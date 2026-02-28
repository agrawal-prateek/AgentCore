"""Tests for context builder recent_actions and summarization improvements."""

from __future__ import annotations

from agent_core.config.agent_config import AgentConfig
from agent_core.context.context_builder import ContextBuilder
from agent_core.context.summarization_policy import DeterministicSummarizationPolicy
from agent_core.state.agent_state import AgentState
from agent_core.state.evidence_graph import EvidenceGraph
from agent_core.state.hypothesis import Hypothesis
from agent_core.state.memory_layers import MidTermMemory
from agent_core.state.stack_tree import StackTree


def _make_state_and_memory() -> tuple[AgentState, MidTermMemory]:
    config = AgentConfig(token_budget=2000, recent_actions_cap=5)
    stack = StackTree(id="stack-1", max_depth=4)
    stack.add_node(node_id="root", objective="Find root cause", parent_id=None)
    evidence = EvidenceGraph(id="ev-1", max_nodes=20)
    memory = MidTermMemory(evidence_graph=evidence, stack_tree=stack, hypotheses={})
    memory.hypotheses["h1"] = Hypothesis(id="h1", description="test hypothesis")
    state = AgentState(
        id="agent-1",
        goal="Find the root cause",
        current_phase="discover",
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


def test_recent_actions_included_in_context() -> None:
    state, memory = _make_state_and_memory()
    builder = ContextBuilder(state.config_snapshot)

    recent_actions = [
        "iter=1:semantic_code_search(query='payment service'):accepted",
        "iter=2:grep_code(pattern='ErrorHandler'):accepted",
        "iter=3:view_file(file_path='src/main.py'):accepted",
    ]

    context = builder.build(
        state=state,
        memory=memory,
        phase_rules="Discover signals",
        latest_tool_result_summary="found 3 results",
        recent_actions=recent_actions,
    )

    assert "recent_actions" in context.payload
    actions = context.payload["recent_actions"]
    assert isinstance(actions, list)
    assert len(actions) == 3
    assert "semantic_code_search" in actions[0]


def test_recent_actions_capped_by_config() -> None:
    config = AgentConfig(token_budget=2000, recent_actions_cap=2)
    state, memory = _make_state_and_memory()
    state.config_snapshot = config
    builder = ContextBuilder(config)

    recent_actions = [
        "iter=1:tool1:accepted",
        "iter=2:tool2:accepted",
        "iter=3:tool3:accepted",
        "iter=4:tool4:accepted",
    ]

    context = builder.build(
        state=state,
        memory=memory,
        phase_rules="Discover signals",
        latest_tool_result_summary="ok",
        recent_actions=recent_actions,
    )

    actions = context.payload["recent_actions"]
    assert isinstance(actions, list)
    assert len(actions) <= 2


def test_recent_actions_empty_when_none() -> None:
    state, memory = _make_state_and_memory()
    builder = ContextBuilder(state.config_snapshot)

    context = builder.build(
        state=state,
        memory=memory,
        phase_rules="Discover signals",
        latest_tool_result_summary="ok",
    )

    assert "recent_actions" in context.payload
    actions = context.payload["recent_actions"]
    assert isinstance(actions, list)
    assert len(actions) == 0


def test_deterministic_summarization_extracts_code_results() -> None:
    policy = DeterministicSummarizationPolicy(max_chars=500)
    payload = {
        "count": 2,
        "snippets": [
            {
                "file_path": "src/payment/handler.py",
                "function_name": "process_payment",
                "content": "def process_payment(amount, currency): ...",
            },
            {
                "file_path": "src/payment/validator.py",
                "class_name": "PaymentValidator",
                "content": "class PaymentValidator: ...",
            },
        ],
    }

    result = policy.summarize(payload)
    assert "payment/handler.py" in result.summary
    assert "process_payment" in result.summary
    assert "list(len=" not in result.summary  # Should NOT produce the old generic format


def test_deterministic_summarization_extracts_file_content() -> None:
    policy = DeterministicSummarizationPolicy(max_chars=500)
    payload = {
        "file_path": "src/main.py",
        "found": True,
        "content_preview": "import flask\napp = flask.Flask(__name__)\n",
    }

    result = policy.summarize(payload)
    assert "src/main.py" in result.summary
    assert "flask" in result.summary


def test_deterministic_summarization_handles_not_found() -> None:
    policy = DeterministicSummarizationPolicy(max_chars=500)
    payload = {
        "file_path": "nonexistent.py",
        "found": False,
        "content_preview": "",
    }

    result = policy.summarize(payload)
    assert "not found" in result.summary


def test_context_includes_latest_tool_payload() -> None:
    state, memory = _make_state_and_memory()
    builder = ContextBuilder(state.config_snapshot)

    payload = {
        "count": 1,
        "snippets": [
            {
                "file_path": "main.py",
                "content": "def hello(): pass",
            }
        ],
    }

    context = builder.build(
        state=state,
        memory=memory,
        phase_rules="Test phase",
        latest_tool_result_summary="found 1 result",
        latest_tool_payload=payload,
    )

    assert "latest_tool_payload" in context.payload
    payload_str = context.payload["latest_tool_payload"]
    assert isinstance(payload_str, str)
    assert "main.py" in payload_str
    assert "def hello(): pass" in payload_str
