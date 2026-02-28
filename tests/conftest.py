from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from agent_core.config.agent_config import AgentConfig
from agent_core.planning.executor import Executor, ExecutorProposal
from agent_core.planning.planner import Planner, PlannerOutput
from agent_core.profiles.profile_interface import CompletionSignal, PhaseDefinition, ProfileInterface
from agent_core.state.agent_state import AgentState
from agent_core.state.evidence_graph import EvidenceGraph
from agent_core.state.hypothesis import Hypothesis
from agent_core.state.memory_layers import MidTermMemory
from agent_core.state.stack_tree import StackTree
from agent_core.tools.tool_registry import Tool, ToolExecutionPayload


class DummyProfile(ProfileInterface):
    def phases(self) -> dict[str, PhaseDefinition]:
        return {
            "discover": PhaseDefinition(
                name="discover",
                description="discover signals",
                allowed_next_phases=("validate",),
                allowed_tools=("echo_tool",),
            ),
            "validate": PhaseDefinition(
                name="validate",
                description="validate signals",
                allowed_next_phases=("synthesize",),
                allowed_tools=("echo_tool",),
            ),
            "synthesize": PhaseDefinition(
                name="synthesize",
                description="synthesize answer",
                allowed_next_phases=(),
                allowed_tools=("echo_tool",),
            ),
        }

    def initial_phase(self) -> str:
        return "discover"

    def system_prompt_template(self) -> str:
        return "system-template"

    def completion_criteria(self, *, phase: str, top_confidence: float) -> CompletionSignal:
        return CompletionSignal(should_complete=phase == "synthesize" and top_confidence > 0.8, reason="")

    def domain_constraints(self) -> list[str]:
        return ["none"]


class StaticPlanner(Planner):
    def __init__(self, output: PlannerOutput) -> None:
        self._output = output

    async def plan(
        self,
        context_payload: dict[str, Any],
        *,
        trace_context=None,
    ) -> PlannerOutput:
        return self._output


class StaticExecutor(Executor):
    def __init__(self, proposal: ExecutorProposal) -> None:
        self._proposal = proposal

    async def propose(
        self,
        context_payload: dict[str, Any],
        objective: str,
        *,
        trace_context=None,
    ) -> ExecutorProposal:
        return self._proposal


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


class EchoTool(Tool):
    name = "echo_tool"
    description = "Echo tool"
    args_schema = EchoArgs

    async def run(self, args: EchoArgs) -> ToolExecutionPayload:  # type: ignore[override]
        return ToolExecutionPayload(content={"echo": args.text}, metadata={"ok": True})


@dataclass
class CoreFixtures:
    config: AgentConfig
    state: AgentState
    memory: MidTermMemory


@pytest.fixture()
def core_fixtures() -> CoreFixtures:
    config = AgentConfig(
        max_iterations=10,
        max_depth=4,
        max_evidence_nodes=10,
        stagnation_threshold=3,
        token_budget=700,
    )

    stack = StackTree(id="stack-1", max_depth=config.max_depth)
    stack.add_node(node_id="root", objective="root objective", parent_id=None)

    evidence = EvidenceGraph(id="graph-1", max_nodes=config.max_evidence_nodes)
    memory = MidTermMemory(evidence_graph=evidence, hypotheses={}, stack_tree=stack)
    memory.hypotheses["h1"] = Hypothesis(id="h1", description="first hypothesis")

    state = AgentState(
        id="agent-1",
        goal="find root cause",
        current_phase="discover",
        config_snapshot=config,
        stack_tree_id="stack-1",
        evidence_graph_id="graph-1",
        hypothesis_set_id="hyp-1",
        working_set_id="ws-1",
        decision_log_id="dl-1",
        summary_index_id="si-1",
        metrics_id="m-1",
    )

    return CoreFixtures(config=config, state=state, memory=memory)
