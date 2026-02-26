from __future__ import annotations

from statistics import mean

from pydantic import BaseModel, ConfigDict, Field


class IterationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration: int = Field(ge=0)
    token_usage: int = Field(ge=0)
    context_tokens: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    branch_count: int = Field(ge=0)
    hypothesis_churn: int = Field(ge=0)
    tool_latency_ms: float = Field(ge=0.0)
    loop_duration_ms: float = Field(ge=0.0)


class MetricsCollector:
    """Collects structured metrics produced by each loop iteration."""

    def __init__(self) -> None:
        self._iterations: list[IterationMetrics] = []

    def record(self, metric: IterationMetrics) -> None:
        self._iterations.append(metric)

    @property
    def iterations(self) -> tuple[IterationMetrics, ...]:
        return tuple(self._iterations)

    def summary(self) -> dict[str, float | int]:
        if not self._iterations:
            return {
                "iterations": 0,
                "avg_token_usage": 0.0,
                "avg_loop_ms": 0.0,
            }
        return {
            "iterations": len(self._iterations),
            "avg_token_usage": mean(m.token_usage for m in self._iterations),
            "avg_loop_ms": mean(m.loop_duration_ms for m in self._iterations),
        }
