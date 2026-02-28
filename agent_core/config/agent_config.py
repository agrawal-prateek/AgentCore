from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SafetyMode(str, Enum):
    READ_ONLY = "read_only"
    STANDARD = "standard"


class AgentConfig(BaseModel):
    """Immutable configuration snapshot for a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: int = Field(default=100, ge=1)
    max_depth: int = Field(default=8, ge=1)
    max_evidence_nodes: int = Field(default=500, ge=10)
    stagnation_threshold: int = Field(default=5, ge=1)
    exploration_bias: float = Field(default=0.5, ge=0.0, le=1.0)
    token_budget: int = Field(default=4000, ge=256)
    planner_model: str = Field(default="planner-default")
    executor_model: str = Field(default="executor-default")
    reproducibility_mode: bool = Field(default=False)
    safety_mode: SafetyMode = Field(default=SafetyMode.STANDARD)
    termination_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    context_parent_summary_cap: int = Field(default=3, ge=1)
    context_hypothesis_cap: int = Field(default=5, ge=1)
    context_evidence_cap: int = Field(default=8, ge=1)
    phase_ancestry_cap: int = Field(default=4, ge=1)
    repeated_tool_call_window: int = Field(default=4, ge=2)
    max_tool_risk_score: float = Field(default=0.7, ge=0.0, le=1.0)
    max_active_agents: int = Field(default=4, ge=1)
    max_spawned_agents_total: int = Field(default=32, ge=1)
    max_agent_depth: int = Field(default=3, ge=1)
