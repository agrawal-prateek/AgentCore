from __future__ import annotations

from agent_core.state.stack_tree import StackNodeStatus, StackTree


def test_stack_collapse_abandons_descendants_and_falls_back_to_parent() -> None:
    tree = StackTree(id="stack-1", max_depth=4)
    tree.add_node(node_id="root", objective="root", parent_id=None)
    tree.add_node(node_id="child", objective="child", parent_id="root")
    tree.add_node(node_id="leaf", objective="leaf", parent_id="child")

    fallback = tree.collapse_branch("child", preserve_ancestor=True)

    assert fallback == "root"
    assert tree.active_node_id == "root"
    assert tree.nodes["leaf"].status == StackNodeStatus.ABANDONED
