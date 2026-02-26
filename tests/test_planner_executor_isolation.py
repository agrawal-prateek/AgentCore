from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_core.context.context_builder import ContextBuilder
from agent_core.context.summarization_policy import DeterministicSummarizationPolicy
from agent_core.engine.decision_engine import DecisionEngine
from agent_core.engine.loop_controller import LoopController
from agent_core.engine.stagnation_detector import StagnationDetector
from agent_core.engine.termination_engine import TerminationEngine
from agent_core.observability.metrics import MetricsCollector
from agent_core.planning.executor import ExecutorProposal
from agent_core.planning.phase_manager import PhaseManager
from agent_core.planning.planner import PlannerOutput
from agent_core.profiles.profile_interface import CompletionSignal, PhaseDefinition, ProfileInterface
from agent_core.state.memory_layers import InMemoryLongTermStorage
from agent_core.tools.sandbox import ToolSandbox
from agent_core.tools.tool_policy import PermissionLevel, ToolPolicy, ToolPolicyEnforcer, ToolPolicyStore
from agent_core.tools.tool_registry import Tool, ToolExecutionPayload, ToolRegistry
from pydantic import BaseModel, ConfigDict, Field


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
        return CompletionSignal(should_complete=False, reason="")

    def domain_constraints(self) -> list[str]:
        return ["none"]


class StaticPlanner:
    def __init__(self, output: PlannerOutput) -> None:
        self._output = output

    async def plan(self, context_payload: dict[str, object]) -> PlannerOutput:
        return self._output


class StaticExecutor:
    def __init__(self, proposal: ExecutorProposal) -> None:
        self._proposal = proposal

    async def propose(self, context_payload: dict[str, object], objective: str) -> ExecutorProposal:
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


def test_executor_proposal_schema_forbids_state_mutation_fields() -> None:
    with pytest.raises(ValidationError):
        ExecutorProposal.model_validate(
            {
                "tool_name": "echo_tool",
                "arguments": {"text": "hello"},
                "expected_outcome": "ok",
                "phase_transition": "synthesize",
            }
        )


@pytest.mark.asyncio
async def test_loop_applies_phase_transition_from_planner_not_executor(core_fixtures) -> None:
    fixtures = core_fixtures
    profile = DummyProfile()

    tool_registry = ToolRegistry()
    tool_registry.register(EchoTool())

    policy_store = ToolPolicyStore()
    policy_store.register(
        "echo_tool",
        ToolPolicy(
            permission_level=PermissionLevel.READ,
            risk_score=0.1,
            allowed_phases=("discover", "validate", "synthesize"),
            max_invocations=5,
        ),
    )

    planner = StaticPlanner(
        PlannerOutput(
            next_objective="probe branch",
            target_branch_id="b1",
            phase_transition="validate",
            reasoning_summary="advance to validation",
            termination_flag=True,
        )
    )
    executor = StaticExecutor(
        ExecutorProposal(tool_name="echo_tool", arguments={"text": "ping"}, expected_outcome="echo")
    )

    controller = LoopController(
        profile=profile,
        planner=planner,
        executor=executor,
        context_builder=ContextBuilder(fixtures.config),
        decision_engine=DecisionEngine(
            tool_registry=tool_registry,
            policy_enforcer=ToolPolicyEnforcer(policy_store=policy_store, max_risk_score=0.8),
            sandbox=ToolSandbox(safety_mode=fixtures.config.safety_mode),
            phase_manager=PhaseManager(profile),
            summarization_policy=DeterministicSummarizationPolicy(),
            long_term_storage=InMemoryLongTermStorage(),
        ),
        phase_manager=PhaseManager(profile),
        stagnation_detector=StagnationDetector(
            threshold=fixtures.config.stagnation_threshold,
            max_depth=fixtures.config.max_depth,
            repeated_tool_window=fixtures.config.repeated_tool_call_window,
        ),
        termination_engine=TerminationEngine(),
        metrics_collector=MetricsCollector(),
    )

    result = await controller.run(state=fixtures.state, memory=fixtures.memory)

    assert result.final_state.current_phase == "validate"
    assert result.final_state.status.value in {"completed", "failed"}
