from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_core.state.agent_state import AgentState
from agent_core.state.memory_layers import MidTermMemory


class TerminationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    should_terminate: bool
    reason: str = Field(min_length=1)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)


class TerminationEngine:
    """Determines loop termination and emits synthesis metadata."""

    def evaluate(
        self,
        *,
        state: AgentState,
        memory: MidTermMemory,
        planner_termination_flag: bool,
        stagnation_counter: int,
        risk_boundary_crossed: bool,
    ) -> TerminationDecision:
        top_conf = max((h.confidence_score for h in memory.hypotheses.values()), default=0.0)

        if top_conf >= state.config_snapshot.termination_confidence_threshold:
            return TerminationDecision(
                should_terminate=True,
                reason="confidence-threshold-reached",
                confidence_score=top_conf,
            )

        if planner_termination_flag:
            return TerminationDecision(
                should_terminate=True,
                reason="planner-signaled-termination",
                confidence_score=top_conf,
            )

        if state.iteration_count >= state.config_snapshot.max_iterations:
            return TerminationDecision(
                should_terminate=True,
                reason="iteration-limit-reached",
                confidence_score=top_conf,
            )

        if stagnation_counter >= state.config_snapshot.stagnation_threshold:
            return TerminationDecision(
                should_terminate=True,
                reason="stagnation-threshold-exceeded",
                confidence_score=top_conf,
            )

        if risk_boundary_crossed:
            return TerminationDecision(
                should_terminate=True,
                reason="risk-boundary-crossed",
                confidence_score=top_conf,
            )

        return TerminationDecision(should_terminate=False, reason="continue", confidence_score=top_conf)
