"""Tests for ES infra-error surface path.

Regression guard for investigation 66311555:
  - ES unreachable → RuntimeError from search_logs
  - DecisionEngine must convert it to a tool-failure outcome (not a crash)
  - LoopController must store the failure summary so the planner sees it
"""
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
from agent_core.profiles.profile_interface import CompletionSignal, PhaseDefinition, ProfileInterface
from agent_core.state.memory_layers import InMemoryLongTermStorage
from agent_core.tools.sandbox import ToolSandbox
from agent_core.tools.tool_policy import PermissionLevel, ToolPolicy, ToolPolicyEnforcer, ToolPolicyStore
from agent_core.tools.tool_registry import Tool, ToolExecutionPayload, ToolRegistry
from conftest import StaticExecutor, StaticPlanner


# ---------------------------------------------------------------------------
# Shared infra
# ---------------------------------------------------------------------------

class SearchLogsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_dsl: dict = Field(default_factory=dict)


class FlakySearchLogsTool(Tool):
    """Simulates search_logs when ES is unreachable (raises RuntimeError)."""

    name = "search_logs"
    description = "Search logs via Elasticsearch"
    args_schema = SearchLogsArgs

    async def run(self, args: SearchLogsArgs) -> ToolExecutionPayload:  # type: ignore[override]
        raise RuntimeError("search_logs infra error: ConnectionError(DNS lookup failed)")


class SearchLogsProfile(ProfileInterface):
    def phases(self) -> dict[str, PhaseDefinition]:
        return {
            "gather": PhaseDefinition(
                name="gather",
                description="gather evidence",
                allowed_next_phases=("gather",),
                allowed_tools=("search_logs",),
            )
        }

    def initial_phase(self) -> str:
        return "gather"

    def planner_prompt_template(self) -> str:
        return "system"

    def executor_prompt_template(self) -> str:
        return "system"

    def completion_criteria(self, *, phase: str, top_confidence: float) -> CompletionSignal:
        return CompletionSignal(should_complete=False, reason="")

    def domain_constraints(self) -> list[str]:
        return []


def _make_engine(fixtures) -> DecisionEngine:
    registry = ToolRegistry()
    registry.register(FlakySearchLogsTool())

    store = ToolPolicyStore()
    store.register(
        "search_logs",
        ToolPolicy(
            permission_level=PermissionLevel.READ,
            risk_score=0.1,
            allowed_phases=("gather",),
            max_invocations=10,
        ),
    )

    profile = SearchLogsProfile()
    return DecisionEngine(
        tool_registry=registry,
        policy_enforcer=ToolPolicyEnforcer(policy_store=store, max_risk_score=0.9),
        sandbox=ToolSandbox(safety_mode=fixtures.config.safety_mode),
        phase_manager=PhaseManager(profile),
        summarization_policy=DeterministicSummarizationPolicy(),
        long_term_storage=InMemoryLongTermStorage(),
    )


# ---------------------------------------------------------------------------
# Test 1: DecisionEngine converts RuntimeError → structured tool-failure outcome
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_es_infra_error_produces_tool_failure_outcome(core_fixtures) -> None:
    """When search_logs raises RuntimeError (ES down), DecisionEngine must:
    - return accepted=False
    - populate tool_summary with 'tool-failure:search_logs'
    - NOT re-raise the exception (crash the loop)
    """
    fixtures = core_fixtures
    fixtures.state.current_phase = "gather"

    engine = _make_engine(fixtures)
    proposal = ExecutorProposal(
        tool_name="search_logs",
        arguments={"query_dsl": {"query": {"match_all": {}}}},
        expected_outcome="log entries",
    )

    outcome = await engine.evaluate_and_execute(
        proposal=proposal,
        context=DecisionContext(current_phase="gather", iteration=1),
        memory=fixtures.memory,
    )

    assert outcome.accepted is False, "ES infra error must not be accepted as success"
    assert outcome.tool_name == "search_logs"
    assert "tool-execution-failed:search_logs" in outcome.reason
    assert outcome.tool_summary is not None
    assert "tool-failure:search_logs" in outcome.tool_summary.summary
    assert "infra error" in outcome.tool_summary.summary


# ---------------------------------------------------------------------------
# Test 2: LoopController stores the tool-failure summary in iteration_summaries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_stores_es_failure_in_iteration_summaries(core_fixtures) -> None:
    """After an ES infra failure, iteration_summaries must contain the error
    message so the planner can read it at the top of its next context window.
    """
    fixtures = core_fixtures
    fixtures.state.current_phase = "gather"

    profile = SearchLogsProfile()
    engine = _make_engine(fixtures)

    planner = StaticPlanner(
        PlannerOutput(
            next_objective="search for login failures",
            target_branch_id="root",
            phase_transition="gather",
            reasoning_summary="gathering logs",
            termination_flag=True,  # terminate after one iteration
        )
    )
    executor = StaticExecutor(
        ExecutorProposal(
            tool_name="search_logs",
            arguments={"query_dsl": {"query": {"match_all": {}}}},
            expected_outcome="log entries",
        )
    )

    controller = LoopController(
        profile=profile,
        planner=planner,
        executor=executor,
        context_builder=ContextBuilder(fixtures.config),
        decision_engine=engine,
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

    # The ES failure summary must appear in iteration_summaries
    assert fixtures.memory.iteration_summaries, "iteration_summaries should not be empty"
    combined = " ".join(fixtures.memory.iteration_summaries)
    assert "tool-failure:search_logs" in combined, (
        f"ES infra failure not found in iteration_summaries: {fixtures.memory.iteration_summaries}"
    )
    assert "infra error" in combined, (
        "Error detail 'infra error' must be present so planner can act on it"
    )
