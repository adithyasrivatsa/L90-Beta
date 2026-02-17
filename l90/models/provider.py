"""Model abstraction layer — swap LLM providers without changing architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelWrapper(ABC):
    """Abstract interface for all LLM model wrappers.

    Every model (Gemini, Llama, future providers) must implement this.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model identifier."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Generate a completion from the model.

        Args:
            prompt: The user/task prompt.
            system_instruction: Optional system-level instruction.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
            response_format: Optional structured output schema hint.

        Returns:
            The generated text response.
        """
        ...

    @abstractmethod
    async def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate a JSON-structured completion.

        The implementation must parse and validate the returned JSON.

        Returns:
            Parsed JSON dict.
        """
        ...


class ModelProvider:
    """Factory that returns the correct model wrapper based on config.

    Design: *Never* hard-code a model anywhere in the codebase.
    Always go through ``ModelProvider.get_worker_model()`` or
    ``ModelProvider.get_manager_model()``.
    """

    _worker_instance: BaseModelWrapper | None = None
    _manager_instance: BaseModelWrapper | None = None

    @classmethod
    def get_worker_model(cls) -> BaseModelWrapper:
        """Return the worker model (currently Gemini 2.5 Flash)."""
        if cls._worker_instance is None:
            from l90.models.gemini_model import GeminiFlashModel
            cls._worker_instance = GeminiFlashModel(role="worker")
        return cls._worker_instance

    @classmethod
    def get_manager_model(cls) -> BaseModelWrapper:
        """Return the manager model (currently Gemini 2.5 Flash, future: Llama 4 Scout)."""
        if cls._manager_instance is None:
            from l90.models.gemini_model import GeminiFlashModel
            cls._manager_instance = GeminiFlashModel(role="manager")
        return cls._manager_instance

    @classmethod
    def reset(cls) -> None:
        """Reset cached instances (useful for testing)."""
        cls._worker_instance = None
        cls._manager_instance = None
