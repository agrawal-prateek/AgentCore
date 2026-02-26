from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agent_core.context.context_builder import ContextBuilder
from agent_core.context.token_budget import estimate_tokens
from agent_core.engine.decision_engine import DecisionContext, DecisionEngine
from agent_core.engine.stagnation_detector import StagnationDetector, StagnationSignal
from agent_core.engine.termination_engine import TerminationDecision, TerminationEngine
from agent_core.observability.metrics import IterationMetrics, MetricsCollector
from agent_core.observability.tracing import get_logger
from agent_core.planning.executor import Executor
from agent_core.planning.phase_manager import PhaseManager
from agent_core.planning.planner import Planner
from agent_core.profiles.profile_interface import ProfileInterface
from agent_core.state.agent_state import AgentState, AgentStatus
from agent_core.state.hypothesis import HypothesisStatus
from agent_core.state.memory_layers import MidTermMemory


@dataclass(frozen=True)
class LoopArtifacts:
    final_state: AgentState
    final_synthesis: dict[str, Any]


class LoopController:
    """Deterministic orchestration loop for stateless reasoning steps."""

    def __init__(
        self,
        *,
        profile: ProfileInterface,
        planner: Planner,
        executor: Executor,
        context_builder: ContextBuilder,
        decision_engine: DecisionEngine,
        phase_manager: PhaseManager,
        stagnation_detector: StagnationDetector,
        termination_engine: TerminationEngine,
        metrics_collector: MetricsCollector,
    ) -> None:
        self._profile = profile
        self._planner = planner
        self._executor = executor
        self._context_builder = context_builder
        self._decision_engine = decision_engine
        self._phase_manager = phase_manager
        self._stagnation_detector = stagnation_detector
        self._termination_engine = termination_engine
        self._metrics = metrics_collector
        self._logger = get_logger("agent_core.loop")

    async def run(self, *, state: AgentState, memory: MidTermMemory) -> LoopArtifacts:
        latest_tool_summary = ""
        termination_reason = "unknown"

        while state.status == AgentStatus.RUNNING:
            loop_start = time.perf_counter()
            self._increment_state(state, "iteration_count", 1, "loop-start")

            phase_def = self._profile.phases()[state.current_phase]
            phase_rules = f"{phase_def.description}; constraints={','.join(self._profile.domain_constraints())}"
            context = self._context_builder.build(
                state=state,
                memory=memory,
                phase_rules=phase_rules,
                latest_tool_result_summary=latest_tool_summary,
            )

            planner_output = await self._planner.plan(context.payload)
            self._log_state_transition(
                state,
                "phase",
                state.current_phase,
                f"planner-objective:{planner_output.next_objective}",
            )

            transition = self._phase_manager.transition(state.current_phase, planner_output.phase_transition)
            if transition.changed:
                self._set_state_field(state, "current_phase", transition.new_phase, transition.reason)

            self._apply_branch_selection(state=state, memory=memory, branch_id=planner_output.target_branch_id, objective=planner_output.next_objective)

            proposal = await self._executor.propose(context.payload, planner_output.next_objective)
            outcome = await self._decision_engine.evaluate_and_execute(
                proposal=proposal,
                context=DecisionContext(current_phase=state.current_phase, iteration=state.iteration_count),
                memory=memory,
            )

            risk_boundary_crossed = outcome.risk_boundary_crossed

            if outcome.accepted and outcome.tool_summary is not None:
                latest_tool_summary = outcome.tool_summary.summary
                self._set_state_field(state, "stagnation_counter", 0, "tool-success")
                memory.add_iteration_summary(outcome.tool_summary.summary)
            else:
                latest_tool_summary = f"rejected:{outcome.reason}"
                self._increment_state(state, "stagnation_counter", 1, "tool-rejected")

            top_conf = max((h.confidence_score for h in memory.hypotheses.values()), default=0.0)
            stagnation_report = self._stagnation_detector.evaluate(
                StagnationSignal(
                    new_evidence_discovered=outcome.accepted,
                    tool_name=proposal.tool_name,
                    tool_args=proposal.arguments,
                    top_hypothesis_confidence=top_conf,
                    branch_depth=state.branch_depth,
                )
            )

            if stagnation_report.triggered:
                self._increment_state(state, "stagnation_counter", 1, "stagnation-detected")
                active = memory.stack_tree.active_node_id
                if active is not None:
                    memory.stack_tree.collapse_branch(active)
                    self._log_state_transition(
                        state,
                        "branch-collapse",
                        memory.stack_tree.active_node_id or "none",
                        ",".join(stagnation_report.reasons),
                    )
                self._force_phase_shift_if_possible(state)

            for hypothesis in memory.hypotheses.values():
                hypothesis.recalculate_confidence(state.iteration_count)

            termination: TerminationDecision = self._termination_engine.evaluate(
                state=state,
                memory=memory,
                planner_termination_flag=planner_output.termination_flag,
                stagnation_counter=state.stagnation_counter,
                risk_boundary_crossed=risk_boundary_crossed,
            )

            loop_ms = (time.perf_counter() - loop_start) * 1000
            self._metrics.record(
                IterationMetrics(
                    iteration=state.iteration_count,
                    token_usage=context.token_count + estimate_tokens(latest_tool_summary),
                    context_tokens=context.token_count,
                    evidence_count=memory.evidence_graph.count,
                    branch_count=len(memory.stack_tree.nodes),
                    hypothesis_churn=self._hypothesis_churn(memory),
                    tool_latency_ms=outcome.tool_latency_ms,
                    loop_duration_ms=loop_ms,
                )
            )

            memory.add_decision(
                {
                    "iteration": state.iteration_count,
                    "planner": planner_output.model_dump(),
                    "executor": proposal.model_dump(),
                    "decision": outcome.model_dump(),
                    "stagnation": stagnation_report.model_dump(),
                    "termination": termination.model_dump(),
                }
            )

            if termination.should_terminate:
                termination_reason = termination.reason
                final_status = (
                    AgentStatus.COMPLETED
                    if termination.reason != "risk-boundary-crossed"
                    else AgentStatus.FAILED
                )
                self._set_state_field(state, "status", final_status, termination.reason)
                break

            if state.iteration_count >= state.config_snapshot.max_iterations:
                termination_reason = "iteration-limit-reached"
                self._set_state_field(state, "status", AgentStatus.FAILED, termination_reason)
                break

        synthesis = self._final_synthesis(memory=memory, reason=termination_reason)
        return LoopArtifacts(final_state=state, final_synthesis=synthesis)

    def _set_state_field(self, state: AgentState, field: str, value: Any, reason: str) -> None:
        old_value = getattr(state, field)
        setattr(state, field, value)
        self._log_state_transition(state, field, value, reason, old_value=old_value)

    def _increment_state(self, state: AgentState, field: str, delta: int, reason: str) -> None:
        old_value = getattr(state, field)
        new_value = old_value + delta
        setattr(state, field, new_value)
        self._log_state_transition(state, field, new_value, reason, old_value=old_value)

    def _log_state_transition(
        self,
        state: AgentState,
        field: str,
        value: Any,
        reason: str,
        *,
        old_value: Any | None = None,
    ) -> None:
        self._logger.info(
            "state_transition",
            extra={
                "event": "state_transition",
                "state": {
                    "agent_id": state.id,
                    "field": field,
                    "old_value": old_value,
                    "new_value": value,
                },
                "details": {"reason": reason, "iteration": state.iteration_count},
            },
        )

    def _apply_branch_selection(
        self,
        *,
        state: AgentState,
        memory: MidTermMemory,
        branch_id: str,
        objective: str,
    ) -> None:
        if branch_id in memory.stack_tree.nodes:
            memory.stack_tree.set_active(branch_id)
        else:
            parent_id = memory.stack_tree.active_node_id
            try:
                memory.stack_tree.add_node(node_id=branch_id, objective=objective, parent_id=parent_id)
            except ValueError:
                if parent_id is not None:
                    memory.stack_tree.collapse_branch(parent_id)
                    self._increment_state(state, "stagnation_counter", 1, "depth-guard-collapse")
        active = memory.stack_tree.active_node_id
        new_depth = memory.stack_tree.nodes[active].depth if active else 0
        self._set_state_field(state, "branch_depth", new_depth, "branch-selection")

        # Update exploration/exploitation accounting deterministically.
        if state.branch_depth > 0:
            self._set_state_field(
                state,
                "exploration_score",
                state.exploration_score + 1.0,
                "branch-depth-positive",
            )
        else:
            self._set_state_field(
                state,
                "exploitation_score",
                state.exploitation_score + 1.0,
                "branch-depth-zero",
            )

    def _force_phase_shift_if_possible(self, state: AgentState) -> None:
        current = self._profile.phases()[state.current_phase]
        if not current.allowed_next_phases:
            return
        next_phase = sorted(current.allowed_next_phases)[0]
        result = self._phase_manager.transition(state.current_phase, next_phase)
        if result.changed:
            self._set_state_field(state, "current_phase", result.new_phase, "stagnation-forced-shift")

    @staticmethod
    def _hypothesis_churn(memory: MidTermMemory) -> int:
        return len(
            [
                h
                for h in memory.hypotheses.values()
                if h.status in {HypothesisStatus.CANDIDATE, HypothesisStatus.STALE}
            ]
        )

    @staticmethod
    def _final_synthesis(memory: MidTermMemory, reason: str) -> dict[str, Any]:
        top_hypotheses = sorted(
            memory.hypotheses.values(),
            key=lambda h: h.confidence_score,
            reverse=True,
        )[:3]
        top_evidence = memory.evidence_graph.top_relevant(5)
        return {
            "termination_reason": reason,
            "top_hypotheses": [h.model_dump() for h in top_hypotheses],
            "evidence_mapping": [e.model_dump() for e in top_evidence],
            "decision_trace_length": len(memory.decision_history),
            "confidence_score": top_hypotheses[0].confidence_score if top_hypotheses else 0.0,
        }
