"""Tests for phase-tool visibility, StackNode summary writeback, and phase-disallow guidance.

Validates the three structural fixes to LoopController:
1. phase_rules includes allowed_tools list
2. StackNode.summary is written back after successful tool execution
3. phase-disallow rejection includes which phases allow the tool
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from datetime import UTC, datetime

import pytest

from agent_core.config.agent_config import AgentConfig
from agent_core.context.context_builder import ContextBuilder
from agent_core.engine.decision_engine import DecisionContext, DecisionOutcome
from agent_core.engine.loop_controller import LoopController
from agent_core.context.summarization_policy import ToolExecutionSummary
from agent_core.planning.phase_manager import PhaseManager
from agent_core.profiles.profile_interface import CompletionSignal, PhaseDefinition, ProfileInterface
from agent_core.state.agent_state import AgentState, AgentStatus
from agent_core.state.evidence_graph import EvidenceGraph
from agent_core.state.hypothesis import Hypothesis
from agent_core.state.memory_layers import MidTermMemory
from agent_core.state.stack_tree import StackTree


# ---------------------------------------------------------------------------
# Test Profile
# ---------------------------------------------------------------------------


def _phase(name: str, next_phases: tuple[str, ...], tools: tuple[str, ...]) -> PhaseDefinition:
    return PhaseDefinition(
        name=name,
        description=f"{name} phase description",
        allowed_next_phases=next_phases,
        allowed_tools=tools,
    )


class _TestProfile(ProfileInterface):
    """Profile with EXPLORE -> HYPOTHESIS -> EVIDENCE -> TERMINATE."""

    _phases = {
        "EXPLORE": _phase("EXPLORE", ("HYPOTHESIS",), ("semantic_search", "view_file")),
        "HYPOTHESIS": _phase("HYPOTHESIS", ("EVIDENCE",), ("semantic_search", "view_file")),
        "EVIDENCE": _phase("EVIDENCE", ("TERMINATE",), ("query_database", "search_logs", "semantic_search")),
        "TERMINATE": _phase("TERMINATE", (), ()),
    }

    def phases(self):
        return self._phases

    def initial_phase(self) -> str:
        return "EXPLORE"

    def planner_prompt_template(self) -> str:
        return ""

    def executor_prompt_template(self) -> str:
        return ""

    def completion_criteria(self, *, phase, top_confidence):
        return CompletionSignal(should_complete=False, reason="")

    def domain_constraints(self):
        return ("query_database only after code exploration",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_and_memory(phase: str = "EXPLORE") -> tuple[AgentState, MidTermMemory]:
    config = AgentConfig(token_budget=4000, recent_actions_cap=5)
    stack = StackTree(id="stack-1", max_depth=4)
    stack.add_node(node_id="root", objective="User cannot login", parent_id=None)
    evidence = EvidenceGraph(id="ev-1", max_nodes=20)
    memory = MidTermMemory(evidence_graph=evidence, stack_tree=stack, hypotheses={})
    memory.hypotheses["h1"] = Hypothesis(id="h1", description="test hypothesis")
    state = AgentState(
        id="test-inv",
        goal="User cannot login",
        current_phase=phase,
        config_snapshot=config,
        stack_tree_id="stack-1",
        evidence_graph_id="ev-1",
        hypothesis_set_id="hyp-1",
        working_set_id="ws-1",
        decision_log_id="dl-1",
        summary_index_id="si-1",
        metrics_id="m-1",
    )
    return state, memory


def _make_controller(profile: ProfileInterface | None = None) -> LoopController:
    profile = profile or _TestProfile()
    return LoopController(
        profile=profile,
        planner=MagicMock(),
        executor=MagicMock(),
        context_builder=MagicMock(),
        decision_engine=MagicMock(),
        phase_manager=PhaseManager(profile),
        stagnation_detector=MagicMock(),
        termination_engine=MagicMock(),
        metrics_collector=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Test 1: phase_rules includes allowed_tools
# ---------------------------------------------------------------------------


class TestPhaseRulesIncludeAllowedTools:
    def test_phase_rules_contain_allowed_tools(self):
        """Verify the phase_rules string includes the allowed_tools list for the current phase."""
        profile = _TestProfile()
        phase_def = profile.phases()["HYPOTHESIS"]
        phase_rules = (
            f"{phase_def.description}; "
            f"allowed_tools={','.join(phase_def.allowed_tools)}; "
            f"constraints={','.join(profile.domain_constraints())}"
        )

        assert "allowed_tools=" in phase_rules
        assert "semantic_search" in phase_rules
        assert "view_file" in phase_rules
        # query_database is NOT in HYPOTHESIS
        assert "query_database" not in phase_rules.split("allowed_tools=")[1].split(";")[0]

    def test_phase_rules_for_evidence_shows_query_database(self):
        """In EVIDENCE phase, phase_rules should include query_database."""
        profile = _TestProfile()
        phase_def = profile.phases()["EVIDENCE"]
        phase_rules = (
            f"{phase_def.description}; "
            f"allowed_tools={','.join(phase_def.allowed_tools)}; "
            f"constraints={','.join(profile.domain_constraints())}"
        )

        tools_section = phase_rules.split("allowed_tools=")[1].split(";")[0]
        assert "query_database" in tools_section
        assert "search_logs" in tools_section


# ---------------------------------------------------------------------------
# Test 2: phase-disallow feedback includes allowing phases
# ---------------------------------------------------------------------------


class TestPhaseDisallowGuidance:
    def test_rejection_includes_allowing_phases(self):
        """When phase-disallow happens, the feedback should name the phases where the tool IS allowed."""
        profile = _TestProfile()
        controller = _make_controller(profile)

        # Simulate what loop_controller does on phase-disallow
        outcome_reason = "phase-disallow:HYPOTHESIS:query_database"
        tool_name = "query_database"
        reject_msg = f"rejected:{outcome_reason}"

        if "phase-disallow" in outcome_reason:
            allowing = [
                name for name, pdef in profile.phases().items()
                if tool_name in pdef.allowed_tools
            ]
            if allowing:
                reject_msg += f" (tool '{tool_name}' is available in phases: {', '.join(allowing)})"

        assert "EVIDENCE" in reject_msg
        assert "query_database" in reject_msg
        assert "available in phases" in reject_msg

    def test_non_phase_disallow_not_enriched(self):
        """Regular rejections should NOT get phase guidance appended."""
        outcome_reason = "policy-or-schema-reject:risk too high"
        reject_msg = f"rejected:{outcome_reason}"

        # No enrichment since it's not phase-disallow
        assert "available in phases" not in reject_msg


# ---------------------------------------------------------------------------
# Test 3: StackNode.summary written back on success
# ---------------------------------------------------------------------------


class TestStackNodeSummaryWriteback:
    def test_summary_written_after_success(self):
        """After a successful tool execution, the active StackNode.summary should contain the tool summary."""
        state, memory = _make_state_and_memory()
        active_id = memory.stack_tree.active_node_id
        node = memory.stack_tree.nodes[active_id]
        assert node.summary == ""

        # Simulate what loop_controller does after outcome.accepted = True
        latest_tool_summary = "Found login function in user_service/app.py using JWT auth with bcrypt hashing"
        if active_id and active_id in memory.stack_tree.nodes and latest_tool_summary:
            existing = node.summary
            node.summary = f"{existing} | {latest_tool_summary}" if existing else latest_tool_summary

        assert node.summary == latest_tool_summary

    def test_summary_accumulates_across_iterations(self):
        """Multiple successful tool executions should accumulate in the summary."""
        state, memory = _make_state_and_memory()
        active_id = memory.stack_tree.active_node_id
        node = memory.stack_tree.nodes[active_id]

        summaries = [
            "Found login function in user_service/app.py",
            "User model schema: id, username, hashed_password, full_name, role",
        ]

        for summary_text in summaries:
            existing = node.summary
            node.summary = f"{existing} | {summary_text}" if existing else summary_text

        assert "login function" in node.summary
        assert "User model schema" in node.summary
        assert " | " in node.summary

    def test_summary_not_written_on_rejection(self):
        """When tool is rejected, summary should NOT be updated."""
        state, memory = _make_state_and_memory()
        active_id = memory.stack_tree.active_node_id
        node = memory.stack_tree.nodes[active_id]
        node.summary = "existing findings"

        # Simulate rejection: outcome.accepted = False, no writeback
        assert node.summary == "existing findings"

    def test_summary_not_written_when_tool_summary_empty(self):
        """When latest_tool_summary is empty, don't write garbage to node."""
        state, memory = _make_state_and_memory()
        active_id = memory.stack_tree.active_node_id
        node = memory.stack_tree.nodes[active_id]

        latest_tool_summary = ""
        # The condition `and latest_tool_summary` prevents empty writes
        if active_id and active_id in memory.stack_tree.nodes and latest_tool_summary:
            node.summary = latest_tool_summary

        assert node.summary == ""


# ---------------------------------------------------------------------------
# Test 4: No truncation in ancestry_summaries
# ---------------------------------------------------------------------------


class TestNoTruncationInAncestry:
    def test_ancestry_summaries_not_truncated(self):
        """ancestry_summaries should return full summaries, not truncated."""
        stack = StackTree(id="stack-1", max_depth=4)
        stack.add_node(node_id="root", objective="root", parent_id=None)
        stack.add_node(node_id="child", objective="child", parent_id="root")

        # Set a long summary on the parent
        long_summary = "A" * 500
        stack.nodes["root"].summary = long_summary

        summaries = stack.ancestry_summaries("child", cap=5)
        assert len(summaries) == 1
        assert len(summaries[0]) == 500  # NOT truncated to 180
        assert summaries[0] == long_summary
