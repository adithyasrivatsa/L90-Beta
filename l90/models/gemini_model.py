"""Gemini 2.5 Flash model wrapper — implements BaseModelWrapper.

Uses asyncio.run_in_executor to make the synchronous google-genai SDK
truly non-blocking inside the async event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from google import genai
from google.genai import types

from l90 import config
from l90.models.provider import BaseModelWrapper

logger = logging.getLogger(__name__)

# Shared thread pool for Gemini calls — avoids creating threads per request
_gemini_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="gemini")


class GeminiFlashModel(BaseModelWrapper):
    """Wrapper around Google Gemini 2.5 Flash via the ``google-genai`` SDK."""

    def __init__(self, role: str = "worker") -> None:
        self._role = role
        self._name = (
            config.MANAGER_MODEL_NAME if role == "manager" else config.WORKER_MODEL_NAME
        )
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)

    # ── BaseModelWrapper interface ─────────────────────────────

    @property
    def model_name(self) -> str:
        return self._name

    def _sync_generate(
        self,
        prompt: str,
        system_instruction: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Synchronous Gemini call — runs in a thread pool."""
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            gen_config.system_instruction = system_instruction

        response = self._client.models.generate_content(
            model=self._name,
            contents=prompt,
            config=gen_config,
        )
        text = response.text or ""
        logger.debug("Gemini [%s] generated %d chars", self._role, len(text))
        return text

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Truly async generate — offloads blocking SDK call to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _gemini_executor,
            partial(
                self._sync_generate,
                prompt,
                system_instruction,
                temperature,
                max_tokens,
            ),
        )

    async def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response.

        Forces the model to return only JSON via system instruction.
        """
        json_system = (
            (system_instruction or "")
            + "\n\nYou MUST respond with valid JSON only. No markdown fences. No extra text."
        ).strip()

        raw = await self.generate(
            prompt,
            system_instruction=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Strip markdown code fences if the model wraps them anyway
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON from Gemini response: %s", exc)
            raise ValueError(f"Model returned invalid JSON: {cleaned[:200]}") from exc
