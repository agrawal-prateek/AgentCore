from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agent_core.context.context_builder import ContextBuilder
from agent_core.context.token_budget import estimate_tokens
from agent_core.engine.decision_engine import DecisionContext, DecisionEngine
from agent_core.engine.stagnation_detector import StagnationDetector, StagnationSignal
from agent_core.engine.termination_engine import TerminationDecision, TerminationEngine
from agent_core.llm.llm_adapter import LLMTraceContext
from agent_core.observability.metrics import IterationMetrics, MetricsCollector
from agent_core.observability.tracing import get_logger
from agent_core.planning.executor import Executor
from agent_core.planning.phase_manager import PhaseManager
from agent_core.planning.planner import Planner
from agent_core.profiles.profile_interface import ProfileInterface
from agent_core.state.agent_tree import AgentNodeStatus, AgentTree
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
        latest_tool_payload: dict[str, Any] | None = None
        termination_reason = "unknown"
        agent_tree = self._ensure_agent_tree(state=state, memory=memory)

        while state.status == AgentStatus.RUNNING:
            loop_start = time.perf_counter()
            self._increment_state(state, "iteration_count", 1, "loop-start")
            try:
                active_agent = agent_tree.select_next_agent(iteration=state.iteration_count)
            except ValueError:
                termination_reason = "no-open-agents"
                self._set_state_field(state, "status", AgentStatus.FAILED, termination_reason)
                break

            phase_def = self._profile.phases()[state.current_phase]
            phase_rules = (
                f"{phase_def.description}; "
                f"allowed_tools={','.join(phase_def.allowed_tools)}; "
                f"constraints={','.join(self._profile.domain_constraints())}"
            )

            # Build recent actions from decision history
            recent_actions = self._build_recent_actions(
                memory.decision_history,
                cap=state.config_snapshot.recent_actions_cap,
            )

            context = self._context_builder.build(
                state=state,
                memory=memory,
                phase_rules=phase_rules,
                latest_tool_result_summary=latest_tool_summary,
                latest_tool_payload=latest_tool_payload,
                recent_actions=recent_actions,
            )

            planner_trace_context = LLMTraceContext(
                investigation_id=state.id,
                iteration=state.iteration_count,
                agent_id=active_agent.id,
                agent_role=active_agent.role,
                task="planner",
            )
            planner_output = await self._planner.plan(context.payload, trace_context=planner_trace_context)
            spawned_children = []
            for spawn_request in planner_output.spawn_children:
                child = agent_tree.spawn_child(
                    parent_agent_id=active_agent.id,
                    request=spawn_request,
                    iteration=state.iteration_count,
                )
                if child is not None:
                    spawned_children.append(child.id)

            self._log_state_transition(
                state,
                "phase",
                state.current_phase,
                f"planner-objective:{planner_output.next_objective}",
            )

            transition = self._phase_manager.transition(state.current_phase, planner_output.phase_transition)
            if transition.changed:
                self._set_state_field(state, "current_phase", transition.new_phase, transition.reason)
            elif transition.disallowed_reason:
                # Surface rejected phase transition to planner on the next iteration
                latest_tool_summary = f"[phase-transition-rejected] {transition.disallowed_reason}"

            # Process hypothesis updates from planner
            for h_id, h_desc in planner_output.hypothesis_update.items():
                if h_id not in memory.hypotheses:
                    from agent_core.state.hypothesis import Hypothesis
                    memory.hypotheses[h_id] = Hypothesis(
                        id=h_id,
                        description=h_desc,
                        last_updated_iteration=state.iteration_count,
                    )

            self._apply_branch_selection(state=state, memory=memory, branch_id=planner_output.target_branch_id, objective=planner_output.next_objective)

            executor_trace_context = LLMTraceContext(
                investigation_id=state.id,
                iteration=state.iteration_count,
                agent_id=active_agent.id,
                agent_role=active_agent.role,
                task="executor",
            )
            proposal = await self._executor.propose(
                context.payload,
                planner_output.next_objective,
                trace_context=executor_trace_context,
            )
            outcome = await self._decision_engine.evaluate_and_execute(
                proposal=proposal,
                context=DecisionContext(
                    current_phase=state.current_phase,
                    iteration=state.iteration_count,
                    agent_id=active_agent.id,
                    investigation_id=state.id,
                    agent_role=active_agent.role,
                ),
                memory=memory,
            )

            risk_boundary_crossed = outcome.risk_boundary_crossed

            if outcome.tool_summary is not None:
                latest_tool_summary = outcome.tool_summary.summary
                memory.add_iteration_summary(outcome.tool_summary.summary)
            if outcome.tool_payload is not None:
                latest_tool_payload = outcome.tool_payload

            if outcome.accepted:
                self._set_state_field(state, "stagnation_counter", 0, "tool-success")
                # Write back accumulated findings to the active branch summary
                active_id = memory.stack_tree.active_node_id
                if active_id and active_id in memory.stack_tree.nodes and latest_tool_summary:
                    node = memory.stack_tree.nodes[active_id]
                    existing = node.summary
                    node.summary = f"{existing} | {latest_tool_summary}" if existing else latest_tool_summary
                # Force immediate termination after conclude tool is accepted
                conclude_requested = proposal.tool_name == "conclude"
                if planner_output.termination_flag or conclude_requested:
                    agent_tree.mark_closed(
                        agent_id=active_agent.id,
                        status=AgentNodeStatus.COMPLETED,
                        iteration=state.iteration_count,
                    )
            else:
                if outcome.tool_summary is None:
                    reject_msg = f"rejected:{outcome.reason}"
                    # Enrich phase-disallow with guidance on which phases allow the tool
                    if "phase-disallow" in outcome.reason:
                        tool = proposal.tool_name
                        allowing = [
                            name for name, pdef in self._profile.phases().items()
                            if tool in pdef.allowed_tools
                        ]
                        if allowing:
                            reject_msg += f" (tool '{tool}' is available in phases: {', '.join(allowing)})"
                    latest_tool_summary = reject_msg
                latest_tool_payload = None
                self._increment_state(state, "stagnation_counter", 1, "tool-rejected")

            top_conf = max((h.confidence_score for h in memory.hypotheses.values()), default=0.0)
            stagnation_signal = StagnationSignal(
                new_evidence_discovered=outcome.accepted,
                tool_name=proposal.tool_name,
                tool_args=proposal.arguments,
                top_hypothesis_confidence=top_conf,
                branch_depth=state.branch_depth,
            )
            stagnation_report = await self._stagnation_detector.evaluate_async(stagnation_signal)

            if stagnation_report.triggered:
                # Only increment for stagnation if tool was accepted (avoid double-count with tool-rejected)
                if outcome.accepted:
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
                # Only force a phase shift once the stagnation counter crosses the
                # configured threshold — not on every individual stagnation event.
                if state.stagnation_counter >= state.config_snapshot.stagnation_threshold:
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
                    "agent": {
                        "id": active_agent.id,
                        "role": active_agent.role,
                        "parent_agent_id": active_agent.parent_agent_id,
                        "depth": active_agent.depth,
                        "spawned_children": spawned_children,
                    },
                    "trace": {
                        "planner_trace_id": getattr(self._planner, "last_trace_id", None),
                        "executor_trace_id": getattr(self._executor, "last_trace_id", None),
                    },
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

    @staticmethod
    def _ensure_agent_tree(*, state: AgentState, memory: MidTermMemory) -> AgentTree:
        if memory.agent_tree is None:
            memory.agent_tree = AgentTree(
                id=f"agents:{state.id}",
                max_active_agents=state.config_snapshot.max_active_agents,
                max_total_agents=state.config_snapshot.max_spawned_agents_total,
                max_depth=state.config_snapshot.max_agent_depth,
            )
        memory.agent_tree.ensure_root(
            agent_id=f"{state.id}:agent:root",
            objective=state.goal,
            role="orchestrator",
        )
        return memory.agent_tree

    @staticmethod
    def _build_recent_actions(decision_history: list[dict[str, Any]], *, cap: int) -> list[str]:
        """Build a bounded list of recent action summaries from decision history."""
        actions: list[str] = []
        start = max(0, len(decision_history) - cap)
        for event in decision_history[start:]:
            iteration = event.get("iteration", "?")
            executor = event.get("executor", {})
            decision = event.get("decision", {})
            tool_name = executor.get("tool_name", "unknown")
            args = executor.get("arguments", {})
            accepted = decision.get("accepted", False)
            status = "accepted" if accepted else "rejected"

            # Build compact args summary (key fields only, truncated)
            args_preview = _compact_args(args)
            actions.append(f"iter={iteration}:{tool_name}({args_preview}):{status}")
        return actions

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
            self._set_state_field(state, "stagnation_counter", 0, "stagnation-reset-on-phase-shift")

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

        # Prefer root_cause from a conclude evidence node if available
        root_cause: str | None = None
        for ev in top_evidence:
            if ev.source_reference == "conclude":
                # The summary contains the rich conclude output
                summary_text = ev.summary or ""
                # Try to extract root_cause from raw evidence data
                raw = memory.evidence_graph.get_raw(ev.id) if hasattr(memory.evidence_graph, "get_raw") else None
                if isinstance(raw, dict) and raw.get("root_cause"):
                    root_cause = raw["root_cause"]
                elif summary_text:
                    root_cause = summary_text
                break

        # Fallback: auto-populate from top hypothesis when conclude was never called
        if not root_cause:
            if top_hypotheses and top_hypotheses[0].confidence_score >= 0.4:
                root_cause = top_hypotheses[0].description
            elif top_hypotheses:
                root_cause = (
                    f"Root cause undetermined — investigation stagnated "
                    f"(confidence: {top_hypotheses[0].confidence_score:.0%})"
                )
            else:
                root_cause = "Root cause undetermined — no hypotheses formed"

        return {
            "termination_reason": reason,
            "root_cause": root_cause,
            "top_hypotheses": [h.model_dump() for h in top_hypotheses],
            "evidence_mapping": [e.model_dump() for e in top_evidence],
            "decision_trace_length": len(memory.decision_history),
            "confidence_score": top_hypotheses[0].confidence_score if top_hypotheses else 0.0,
            "agent_count": memory.agent_tree.total_count if memory.agent_tree is not None else 0,
        }


def _compact_args(args: dict[str, Any], max_len: int = 80) -> str:
    """Build a compact one-line preview of tool arguments."""
    parts: list[str] = []
    for key in ("query", "q", "file_path", "pattern", "index", "collection", "sql"):
        val = args.get(key)
        if val is not None:
            text = str(val)
            if len(text) > 50:
                text = text[:47] + "..."
            parts.append(f"{key}='{text}'")
    preview = ", ".join(parts)
    if not preview:
        preview = ", ".join(f"{k}={v}" for k, v in list(args.items())[:2])
    return preview[:max_len]
