from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from agent_core.config.agent_config import AgentConfig


class AgentStatus(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentState(BaseModel):
    """Strict root state contract for a running agent."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    status: AgentStatus = Field(default=AgentStatus.RUNNING)
    current_phase: str = Field(min_length=1)
    iteration_count: int = Field(default=0, ge=0)
    branch_depth: int = Field(default=0, ge=0)
    stagnation_counter: int = Field(default=0, ge=0)
    exploration_score: float = Field(default=0.0, ge=0.0)
    exploitation_score: float = Field(default=0.0, ge=0.0)
    config_snapshot: AgentConfig

    stack_tree_id: str = Field(min_length=1)
    evidence_graph_id: str = Field(min_length=1)
    hypothesis_set_id: str = Field(min_length=1)

    working_set_id: str = Field(min_length=1)
    decision_log_id: str = Field(min_length=1)
    summary_index_id: str = Field(min_length=1)
    metrics_id: str = Field(min_length=1)
