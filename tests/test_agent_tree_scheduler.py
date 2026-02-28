from __future__ import annotations

from agent_core.state.agent_tree import AgentNodeStatus, AgentSpawnRequest, AgentTree


def test_agent_tree_enforces_spawn_caps() -> None:
    tree = AgentTree(id="agents:1", max_active_agents=2, max_total_agents=3, max_depth=2)
    root = tree.ensure_root(agent_id="root", objective="root objective")
    assert root.id == "root"

    child_one = tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="child one", role="specialist", priority=0.9, child_id="c1"),
        iteration=1,
    )
    assert child_one is not None

    # Open cap reached (root + c1 = 2), should not spawn another open child.
    blocked = tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="child two", role="specialist", priority=0.8, child_id="c2"),
        iteration=2,
    )
    assert blocked is None

    tree.mark_closed(agent_id="c1", status=AgentNodeStatus.COMPLETED, iteration=2)
    child_two = tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="child two", role="specialist", priority=0.8, child_id="c2"),
        iteration=3,
    )
    assert child_two is not None


def test_agent_tree_scheduler_is_deterministic() -> None:
    tree = AgentTree(id="agents:2", max_active_agents=4, max_total_agents=6, max_depth=3)
    tree.ensure_root(agent_id="root", objective="root")
    tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="high prio", role="specialist", priority=0.9, child_id="a"),
        iteration=1,
    )
    tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="same prio later", role="specialist", priority=0.9, child_id="b"),
        iteration=1,
    )
    tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="low prio", role="specialist", priority=0.7, child_id="c"),
        iteration=1,
    )

    # Highest priority first; ties broken by created_seq then id.
    assert tree.select_next_agent(iteration=2).id == "a"
