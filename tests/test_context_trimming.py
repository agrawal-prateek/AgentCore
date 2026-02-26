from __future__ import annotations

from agent_core.config.agent_config import AgentConfig
from agent_core.context.context_builder import ContextBuilder
from agent_core.state.agent_state import AgentState
from agent_core.state.evidence_graph import EvidenceGraph, EvidenceNode
from agent_core.state.hypothesis import Hypothesis
from agent_core.state.memory_layers import MidTermMemory
from agent_core.state.stack_tree import StackTree


def test_context_builder_enforces_token_budget_with_trimming() -> None:
    config = AgentConfig(token_budget=620, context_hypothesis_cap=8, context_evidence_cap=8)
    stack = StackTree(id="stack-1", max_depth=4)
    stack.add_node(node_id="root", objective="Investigate the issue deeply" * 4, parent_id=None)

    evidence = EvidenceGraph(id="ev-1", max_nodes=20)
    for i in range(10):
        evidence.add_or_merge_node(
            EvidenceNode(
                id=f"e{i}",
                type="summary",
                source_reference="tool",
                summary=("evidence details " * 20) + str(i),
                raw_pointer=f"raw:{i}",
                relevance_score=1 - (i * 0.05),
                weight=0.6,
                created_iteration=i,
            ),
            current_iteration=i,
        )

    memory = MidTermMemory(evidence_graph=evidence, stack_tree=stack, hypotheses={})
    for i in range(10):
        memory.hypotheses[f"h{i}"] = Hypothesis(id=f"h{i}", description=("hypothesis text " * 18) + str(i))

    state = AgentState(
        id="agent-1",
        goal="Find the root cause with strict budget controls",
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

    context = ContextBuilder(config).build(
        state=state,
        memory=memory,
        phase_rules="Discover and prioritize high value evidence",
        latest_tool_result_summary="latest output " * 40,
    )

    assert context.token_count <= config.token_budget - 512
    assert context.trimmed is True
    assert "goal" in context.payload
