from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from agent_core.context.context_builder import ContextBuilder
from agent_core.context.summarization_policy import DeterministicSummarizationPolicy
from agent_core.engine.decision_engine import DecisionContext, DecisionEngine
from agent_core.engine.loop_controller import LoopController
from agent_core.engine.stagnation_detector import StagnationDetector
from agent_core.engine.termination_engine import TerminationEngine
from agent_core.observability.metrics import MetricsCollector
from agent_core.planning.executor import ExecutorProposal
from agent_core.planning.phase_manager import PhaseManager
from agent_core.planning.planner import PlannerOutput
from agent_core.state.memory_layers import InMemoryLongTermStorage
from agent_core.tools.sandbox import ToolSandbox
from agent_core.tools.tool_policy import PermissionLevel, ToolPolicy, ToolPolicyEnforcer, ToolPolicyStore
from agent_core.tools.tool_registry import Tool, ToolExecutionPayload, ToolRegistry
from agent_core.profiles.profile_interface import CompletionSignal, PhaseDefinition, ProfileInterface
from conftest import StaticExecutor, StaticPlanner


class BrokenArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


class BrokenTool(Tool):
    name = "broken_tool"
    description = "Always fails"
    args_schema = BrokenArgs

    async def run(self, args: BrokenArgs) -> ToolExecutionPayload:  # type: ignore[override]
        raise RuntimeError("backend unavailable")


class BrokenProfile(ProfileInterface):
    def phases(self) -> dict[str, PhaseDefinition]:
        return {
            "discover": PhaseDefinition(
                name="discover",
                description="discover",
                allowed_next_phases=("discover",),
                allowed_tools=("broken_tool",),
            )
        }

    def initial_phase(self) -> str:
        return "discover"

    def planner_prompt_template(self) -> str:
        return "system"

    def executor_prompt_template(self) -> str:
        return "system"

    def completion_criteria(self, *, phase: str, top_confidence: float) -> CompletionSignal:
        return CompletionSignal(should_complete=False, reason="")

    def domain_constraints(self) -> list[str]:
        return []


@pytest.mark.asyncio
async def test_decision_engine_returns_structured_outcome_for_tool_failure(core_fixtures) -> None:
    fixtures = core_fixtures

    registry = ToolRegistry()
    registry.register(BrokenTool())

    policies = ToolPolicyStore()
    policies.register(
        "broken_tool",
        ToolPolicy(
            permission_level=PermissionLevel.READ,
            risk_score=0.1,
            allowed_phases=("discover",),
            max_invocations=5,
        ),
    )

    engine = DecisionEngine(
        tool_registry=registry,
        policy_enforcer=ToolPolicyEnforcer(policy_store=policies, max_risk_score=0.8),
        sandbox=ToolSandbox(safety_mode=fixtures.config.safety_mode),
            phase_manager=PhaseManager(BrokenProfile()),
        summarization_policy=DeterministicSummarizationPolicy(),
        long_term_storage=InMemoryLongTermStorage(),
    )

    outcome = await engine.evaluate_and_execute(
        proposal=ExecutorProposal(tool_name="broken_tool", arguments={"text": "x"}, expected_outcome="none"),
        context=DecisionContext(current_phase="discover", iteration=1),
        memory=fixtures.memory,
    )

    assert outcome.accepted is False
    assert outcome.tool_name == "broken_tool"
    assert outcome.reason.startswith("tool-execution-failed:broken_tool")
    assert outcome.tool_summary is not None
    assert "tool-failure:broken_tool" in outcome.tool_summary.summary


@pytest.mark.asyncio
async def test_loop_keeps_tool_failure_summary_in_memory(core_fixtures) -> None:
    fixtures = core_fixtures
    profile = BrokenProfile()

    registry = ToolRegistry()
    registry.register(BrokenTool())

    policies = ToolPolicyStore()
    policies.register(
        "broken_tool",
        ToolPolicy(
            permission_level=PermissionLevel.READ,
            risk_score=0.1,
            allowed_phases=("discover", "validate", "synthesize"),
            max_invocations=5,
        ),
    )

    planner = StaticPlanner(
        PlannerOutput(
            next_objective="try a tool",
            target_branch_id="b1",
            phase_transition="discover",
            reasoning_summary="attempt tool",
            termination_flag=True,
        )
    )
    executor = StaticExecutor(
        ExecutorProposal(tool_name="broken_tool", arguments={"text": "x"}, expected_outcome="none")
    )

    controller = LoopController(
        profile=profile,
        planner=planner,
        executor=executor,
        context_builder=ContextBuilder(fixtures.config),
        decision_engine=DecisionEngine(
            tool_registry=registry,
            policy_enforcer=ToolPolicyEnforcer(policy_store=policies, max_risk_score=0.8),
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

    await controller.run(state=fixtures.state, memory=fixtures.memory)

    assert fixtures.memory.iteration_summaries
    assert any("tool-failure:broken_tool" in text for text in fixtures.memory.iteration_summaries)
