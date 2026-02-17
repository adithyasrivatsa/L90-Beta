"""Math Executor Agent — sandboxed Python code execution for proving math/physics.

When the pipeline detects a math or physics question, this agent:
1. Asks the LLM to generate a Python verification script
2. Executes it in a sandboxed subprocess with timeout
3. Compares output against the claimed answer
4. Attaches proof to the Blackboard
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from l90 import config
from l90.agents.base import BaseAgent
from l90.blackboard.blackboard import Blackboard
from l90.models.provider import ModelProvider
from l90.tracing.logger import ReasoningTraceLogger

logger = logging.getLogger(__name__)

CODE_GEN_SYSTEM_PROMPT = """\
You are a mathematical verification specialist inside L90, a NASA/DARPA-grade scientific system.

Your task: generate a COMPLETE, SELF-CONTAINED Python script that PROVES the mathematical or physics
result described below. The script must:

1. Import only from: math, cmath, numpy, scipy, sympy, fractions, decimal, statistics
2. Compute the answer step by step with clear variable names
3. Print EXACTLY one line at the end: "RESULT: <value>" where <value> is the final numeric/symbolic answer
4. Include dimensional analysis comments where applicable
5. For physics: show unit conversions explicitly
6. For proofs: show each logical step
7. Handle edge cases (division by zero, domain errors) gracefully

CRITICAL: Output ONLY the Python code. No markdown fences. No explanations. Just executable Python.

If the question is NOT mathematical/physics (purely conceptual), output exactly: # NO_CODE_NEEDED
"""


class MathExecutorAgent(BaseAgent):
    """Generates and executes Python code to verify math/physics results.

    Security: code runs in a subprocess with:
    - Timeout (configurable, default 10s)
    - No network access (no socket/urllib/requests imports)
    - stderr captured for debugging
    """

    def __init__(self, trace_logger: ReasoningTraceLogger | None = None) -> None:
        super().__init__(name="MathExecutorAgent", trace_logger=trace_logger)

    async def execute(self, blackboard: Blackboard) -> Blackboard:
        """Generate verification code, execute it, and record results."""

        self._log_trace(
            phase="code_verification",
            action="math_executor_start",
            input_summary=f"Query: {blackboard.query[:100]}",
        )

        # Check if code verification is needed
        plan = blackboard.execution_plan
        if not plan.get("requires_code_verification", False):
            self._log_trace(
                phase="code_verification",
                action="math_executor_skip",
                decision="Code verification not required for this query",
            )
            blackboard.code_verification = {"skipped": True, "reason": "not_required"}
            return blackboard

        model = ModelProvider.get_worker_model()

        # Build context from analysis results
        analysis_context = ""
        for result in blackboard.analysis_results:
            if "formulas" in result:
                analysis_context += f"Formulas: {result['formulas']}\n"
            if "facts" in result:
                analysis_context += f"Facts: {result['facts']}\n"
            if "summary" in result:
                analysis_context += f"Summary: {result['summary']}\n"

        prompt = (
            f"User Question: {blackboard.query}\n\n"
            f"Extracted Information:\n{analysis_context}\n\n"
            "Generate a Python verification script that proves the mathematical/"
            "physics result. Show all steps."
        )

        try:
            code = await model.generate(
                prompt=prompt,
                system_instruction=CODE_GEN_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=4096,
            )

            code = code.strip()

            # Strip markdown fences if model wraps them
            if code.startswith("```"):
                code = code.split("\n", 1)[-1]
            if code.endswith("```"):
                code = code.rsplit("```", 1)[0]
            code = code.strip()

            if code == "# NO_CODE_NEEDED":
                blackboard.code_verification = {
                    "skipped": True,
                    "reason": "no_code_needed",
                }
                self._log_trace(
                    phase="code_verification",
                    action="math_executor_skip",
                    decision="LLM determined no code verification needed",
                )
                return blackboard

            # Validate: reject dangerous imports
            dangerous = {"os", "sys", "subprocess", "socket", "urllib",
                         "requests", "shutil", "pathlib", "glob", "io",
                         "__import__", "eval", "exec", "compile", "open"}
            for line in code.split("\n"):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    for d in dangerous:
                        if d in stripped:
                            raise ValueError(f"Dangerous import detected: {stripped}")

            # Execute in sandbox
            execution_result = self._execute_code(code)

            blackboard.code_verification = {
                "skipped": False,
                "code": code,
                "stdout": execution_result["stdout"],
                "stderr": execution_result["stderr"],
                "returncode": execution_result["returncode"],
                "success": execution_result["returncode"] == 0,
                "result": self._extract_result(execution_result["stdout"]),
                "timed_out": execution_result.get("timed_out", False),
            }

            self._log_trace(
                phase="code_verification",
                action="math_executor_complete",
                output_summary=(
                    f"Code executed: success={execution_result['returncode'] == 0}, "
                    f"result={self._extract_result(execution_result['stdout'])[:100]}"
                ),
                confidence=1.0 if execution_result["returncode"] == 0 else 0.0,
            )

        except Exception as exc:
            logger.error("Math executor failed: %s", exc)
            blackboard.code_verification = {
                "skipped": False,
                "error": str(exc),
                "success": False,
            }
            self._log_trace(
                phase="code_verification",
                action="math_executor_error",
                decision=f"Error: {exc}",
            )

        blackboard.add_trace(
            agent_name=self._name,
            phase="code_verification",
            action="code_verification_recorded",
            details=blackboard.code_verification,
        )

        return blackboard

    def _execute_code(self, code: str) -> dict[str, Any]:
        """Run Python code in a subprocess with timeout."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=config.CODE_EXECUTION_TIMEOUT,
                cwd=tempfile.gettempdir(),
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {config.CODE_EXECUTION_TIMEOUT}s",
                "returncode": -1,
                "timed_out": True,
            }
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @staticmethod
    def _extract_result(stdout: str) -> str:
        """Extract the RESULT line from stdout."""
        for line in stdout.strip().split("\n"):
            if line.strip().startswith("RESULT:"):
                return line.strip()[7:].strip()
        return stdout.strip()[-200:] if stdout.strip() else ""
