from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_core.context.token_budget import estimate_tokens


class ToolExecutionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    entities: tuple[str, ...] = Field(default_factory=tuple)
    compression_score: float = Field(ge=0.0, le=1.0)


class SummarizationPolicy(ABC):
    @abstractmethod
    def summarize(self, payload: dict[str, Any]) -> ToolExecutionSummary:
        raise NotImplementedError


class DeterministicSummarizationPolicy(SummarizationPolicy):
    """Produces a fixed-shape summary and lightweight entity extraction."""

    def __init__(self, max_chars: int = 320) -> None:
        self._max_chars = max_chars

    def summarize(self, payload: dict[str, Any]) -> ToolExecutionSummary:
        flattened = "; ".join(f"{k}={self._to_scalar(v)}" for k, v in sorted(payload.items()))
        trimmed = flattened[: self._max_chars].strip() or "empty-tool-output"

        entity_candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", trimmed)
        entities = tuple(dict.fromkeys(entity_candidates[:8]))

        source_tokens = estimate_tokens(str(payload))
        summary_tokens = max(1, estimate_tokens(trimmed))
        compression = min(1.0, round(summary_tokens / source_tokens, 4)) if source_tokens > 0 else 1.0
        return ToolExecutionSummary(summary=trimmed, entities=entities, compression_score=compression)

    @staticmethod
    def _to_scalar(value: Any) -> str:
        if isinstance(value, (int, float, bool, str)):
            return str(value)
        if value is None:
            return "null"
        if isinstance(value, dict):
            return f"dict(len={len(value)})"
        if isinstance(value, list):
            return f"list(len={len(value)})"
        return type(value).__name__
