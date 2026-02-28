from __future__ import annotations

import time
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from agent_core.context.summarization_policy import SummarizationPolicy, ToolExecutionSummary
from agent_core.planning.executor import ExecutorProposal
from agent_core.planning.phase_manager import PhaseManager
from agent_core.state.evidence_graph import EvidenceNode
from agent_core.state.memory_layers import LongTermStoragePort, MidTermMemory
from agent_core.tools.sandbox import ToolSandbox
from agent_core.tools.tool_policy import ToolPolicyEnforcer
from agent_core.tools.tool_registry import ToolRegistry


class DecisionOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    reason: str = Field(min_length=1)
    tool_name: str | None = None
    tool_summary: ToolExecutionSummary | None = None
    evidence_node_id: str | None = None
    tool_artifact_id: str | None = None
    tool_latency_ms: float = Field(default=0.0, ge=0.0)
    risk_boundary_crossed: bool = False


@dataclass(frozen=True)
class DecisionContext:
    current_phase: str
    iteration: int
    agent_id: str = "unknown-agent"
    confirmation_granted: bool = False


class DecisionEngine:
    """Mediates executor proposals and enforces tool validation + sandboxing."""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        policy_enforcer: ToolPolicyEnforcer,
        sandbox: ToolSandbox,
        phase_manager: PhaseManager,
        summarization_policy: SummarizationPolicy,
        long_term_storage: LongTermStoragePort,
    ) -> None:
        self._tool_registry = tool_registry
        self._policy_enforcer = policy_enforcer
        self._sandbox = sandbox
        self._phase_manager = phase_manager
        self._summarization_policy = summarization_policy
        self._long_term_storage = long_term_storage
        self._invocation_counts: dict[str, int] = {}

    async def evaluate_and_execute(
        self,
        *,
        proposal: ExecutorProposal,
        context: DecisionContext,
        memory: MidTermMemory,
    ) -> DecisionOutcome:
        if not self._tool_registry.exists(proposal.tool_name):
            return DecisionOutcome(accepted=False, reason=f"unknown-tool:{proposal.tool_name}")

        if not self._phase_manager.can_use_tool(context.current_phase, proposal.tool_name):
            return DecisionOutcome(
                accepted=False,
                reason=f"phase-disallow:{context.current_phase}:{proposal.tool_name}",
            )

        invocation_count = self._invocation_counts.get(proposal.tool_name, 0)
        try:
            policy = self._policy_enforcer.validate(
                tool_name=proposal.tool_name,
                current_phase=context.current_phase,
                invocation_count=invocation_count,
                confirmation_granted=context.confirmation_granted,
            )
            self._sandbox.assert_allowed(policy)
            args = self._tool_registry.validate_arguments(proposal.tool_name, proposal.arguments)
        except PermissionError as exc:
            return DecisionOutcome(
                accepted=False,
                reason=f"policy-or-schema-reject:{exc}",
                risk_boundary_crossed=("risk policy" in str(exc)),
            )
        except (ValueError, KeyError) as exc:
            return DecisionOutcome(accepted=False, reason=f"policy-or-schema-reject:{exc}")

        tool = self._tool_registry.get(proposal.tool_name)
        start = time.perf_counter()
        try:
            payload = await tool.run(args)
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            self._invocation_counts[proposal.tool_name] = invocation_count + 1
            failure_message = f"{exc.__class__.__name__}: {exc}"
            artifact_id = await self._long_term_storage.store_tool_output(
                {
                    "tool": proposal.tool_name,
                    "args": proposal.arguments,
                    "agent_id": context.agent_id,
                    "status": "failed",
                    "error": failure_message,
                    "raw": {},
                    "metadata": {},
                    "iteration": context.iteration,
                }
            )
            failure_summary = ToolExecutionSummary(
                summary=f"tool-failure:{proposal.tool_name}:{failure_message}"[:320],
                entities=(proposal.tool_name, "tool_failure", exc.__class__.__name__),
                compression_score=1.0,
            )
            return DecisionOutcome(
                accepted=False,
                reason=f"tool-execution-failed:{proposal.tool_name}:{failure_message}",
                tool_name=proposal.tool_name,
                tool_summary=failure_summary,
                tool_artifact_id=artifact_id,
                tool_latency_ms=latency,
            )
        latency = (time.perf_counter() - start) * 1000

        raw_pointer = await self._long_term_storage.store_tool_output(
            {
                "tool": proposal.tool_name,
                "args": proposal.arguments,
                "agent_id": context.agent_id,
                "status": "success",
                "raw": payload.content,
                "metadata": payload.metadata,
                "iteration": context.iteration,
            }
        )

        summary = self._summarization_policy.summarize(payload.content)
        evidence_id = f"ev-{context.iteration}-{len(memory.evidence_graph.nodes)}"
        evidence_node = EvidenceNode(
            id=evidence_id,
            type="summary",
            source_reference=proposal.tool_name,
            summary=summary.summary,
            raw_pointer=raw_pointer,
            relevance_score=0.7,
            weight=0.65,
            created_iteration=context.iteration,
        )
        actual_id = memory.evidence_graph.add_or_merge_node(
            evidence_node,
            current_iteration=context.iteration,
        )

        self._invocation_counts[proposal.tool_name] = invocation_count + 1
        return DecisionOutcome(
            accepted=True,
            reason="executed",
            tool_name=proposal.tool_name,
            tool_summary=summary,
            evidence_node_id=actual_id,
            tool_artifact_id=raw_pointer,
            tool_latency_ms=latency,
        )
