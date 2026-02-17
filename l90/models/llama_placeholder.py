"""Llama 4 Scout placeholder — future manager model upgrade."""

from __future__ import annotations

from typing import Any

from l90.models.provider import BaseModelWrapper


class Llama4ScoutManager(BaseModelWrapper):
    """Placeholder for Llama 4 Scout manager model.

    All methods raise ``NotImplementedError``.
    Swap into ``ModelProvider`` when the model is available.
    """

    @property
    def model_name(self) -> str:
        return "llama-4-scout"

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError("Llama 4 Scout is not yet available.")

    async def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        raise NotImplementedError("Llama 4 Scout is not yet available.")

    # ── Scout-specific planning interface (future) ─────────────

    async def plan(self) -> dict[str, Any]:
        """Generate an execution plan."""
        raise NotImplementedError

    async def orchestrate(self) -> None:
        """Run the orchestration loop."""
        raise NotImplementedError

    async def analyze_blackboard(self) -> dict[str, Any]:
        """Analyze current Blackboard state."""
        raise NotImplementedError

    async def spawn_agents(self) -> list[str]:
        """Determine and spawn required agents."""
        raise NotImplementedError
