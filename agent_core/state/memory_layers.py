from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_core.state.evidence_graph import EvidenceGraph
from agent_core.state.hypothesis import Hypothesis
from agent_core.state.agent_tree import AgentTree
from agent_core.state.stack_tree import StackTree


class MidTermMemory(BaseModel):
    """Structured memory queried selectively by the context builder."""

    model_config = ConfigDict(extra="forbid")

    evidence_graph: EvidenceGraph
    hypotheses: dict[str, Hypothesis] = Field(default_factory=dict)
    stack_tree: StackTree
    phase_summaries: dict[str, str] = Field(default_factory=dict)
    decision_history: list[dict[str, Any]] = Field(default_factory=list)
    iteration_summaries: list[str] = Field(default_factory=list)
    agent_tree: AgentTree | None = None

    def add_decision(self, decision: dict[str, Any]) -> None:
        self.decision_history.append(decision)

    def add_iteration_summary(self, summary: str) -> None:
        self.iteration_summaries.append(summary)

    def top_hypotheses(self, limit: int) -> list[Hypothesis]:
        ranked = sorted(
            self.hypotheses.values(),
            key=lambda h: (h.confidence_score, -h.last_updated_iteration),
            reverse=True,
        )
        return ranked[:limit]


class LongTermMemory(BaseModel):
    """Raw outputs and artifacts, never directly fed back into working memory."""

    model_config = ConfigDict(extra="forbid")

    raw_tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    raw_reasoning_logs: list[str] = Field(default_factory=list)
    snapshots: list[dict[str, Any]] = Field(default_factory=list)


class LongTermStoragePort(ABC):
    @abstractmethod
    async def store_tool_output(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def store_snapshot(self, payload: dict[str, Any]) -> str:
        raise NotImplementedError


class InMemoryLongTermStorage(LongTermStoragePort):
    def __init__(self) -> None:
        self._outputs: list[dict[str, Any]] = []
        self._snapshots: list[dict[str, Any]] = []

    async def store_tool_output(self, payload: dict[str, Any]) -> str:
        self._outputs.append(payload)
        return f"tool-output:{len(self._outputs) - 1}"

    async def store_snapshot(self, payload: dict[str, Any]) -> str:
        self._snapshots.append(payload)
        return f"snapshot:{len(self._snapshots) - 1}"
