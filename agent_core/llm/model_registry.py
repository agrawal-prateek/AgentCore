from __future__ import annotations

from agent_core.llm.llm_adapter import LLMAdapter


class ModelRegistry:
    """Registry of swappable LLM adapters."""

    def __init__(self) -> None:
        self._models: dict[str, LLMAdapter] = {}

    def register(self, model_name: str, adapter: LLMAdapter) -> None:
        self._models[model_name] = adapter

    def get(self, model_name: str) -> LLMAdapter:
        try:
            return self._models[model_name]
        except KeyError as exc:
            raise KeyError(f"Model '{model_name}' is not registered") from exc
