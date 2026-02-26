from __future__ import annotations

from agent_core.state.evidence_graph import EvidenceGraph, EvidenceNode


def test_evidence_prunes_low_weight_when_over_limit() -> None:
    graph = EvidenceGraph(id="g1", max_nodes=3)

    nodes = [
        EvidenceNode(
            id=f"e{i}",
            type="log",
            source_reference="src",
            summary=f"summary {i}",
            raw_pointer=f"raw:{i}",
            relevance_score=0.9 - (i * 0.1),
            weight=0.1 + (i * 0.2),
            created_iteration=i,
        )
        for i in range(5)
    ]

    for i, node in enumerate(nodes):
        graph.add_or_merge_node(node, current_iteration=i)

    assert graph.count == 3
    assert "e0" not in graph.nodes
    assert "e1" not in graph.nodes
    assert "e4" in graph.nodes


def test_evidence_deduplicates_by_hash_and_merges_weight() -> None:
    graph = EvidenceGraph(id="g2", max_nodes=5)

    first = EvidenceNode(
        id="e1",
        type="summary",
        source_reference="tool-a",
        summary="same observation",
        raw_pointer="raw:1",
        relevance_score=0.6,
        weight=0.5,
        created_iteration=1,
    )
    duplicate = EvidenceNode(
        id="e2",
        type="summary",
        source_reference="tool-a",
        summary="same observation",
        raw_pointer="raw:2",
        relevance_score=0.9,
        weight=0.7,
        created_iteration=2,
    )

    first_id = graph.add_or_merge_node(first, current_iteration=1)
    second_id = graph.add_or_merge_node(duplicate, current_iteration=2)

    assert first_id == "e1"
    assert second_id == "e1"
    assert graph.count == 1
    assert graph.nodes["e1"].weight > 0.5
    assert graph.nodes["e1"].relevance_score == 0.9
