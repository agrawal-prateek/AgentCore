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

    # All agents have last_active_iteration=0, so ties break by priority, then depth, then created_seq.
    # root(prio=0.5,d=0), a(prio=0.9,d=1), b(prio=0.9,d=1), c(prio=0.7,d=1)
    # Among those at last_active=0: highest priority first → "a" (prio=0.9, earlier seq)
    assert tree.select_next_agent(iteration=2).id == "a"


def test_agent_tree_scheduler_round_robin() -> None:
    """Verify agents take turns and parents wait for children."""
    tree = AgentTree(id="agents:3", max_active_agents=4, max_total_agents=6, max_depth=3)
    tree.ensure_root(agent_id="root", objective="root")
    tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="auth specialist", role="specialist", priority=0.8, child_id="s1"),
        iteration=1,
    )
    tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="infra specialist", role="specialist", priority=0.7, child_id="s2"),
        iteration=1,
    )

    # Root has open children → excluded from scheduling.
    # First pick: s1 wins (higher prio among children at last_active=0)
    picked_1 = tree.select_next_agent(iteration=2)
    assert picked_1.id == "s1"

    # Second pick: s2 (round-robin, s1 just went)
    picked_2 = tree.select_next_agent(iteration=3)
    assert picked_2.id == "s2"

    # Third pick: s1 again (round-robin between the two children, root still excluded)
    picked_3 = tree.select_next_agent(iteration=4)
    assert picked_3.id == "s1"

    # Close both children — root becomes eligible again
    tree.mark_closed(agent_id="s1", status=AgentNodeStatus.COMPLETED, iteration=4)
    tree.mark_closed(agent_id="s2", status=AgentNodeStatus.COMPLETED, iteration=4)

    # Now root is the only open agent, it gets scheduled to synthesize
    picked_4 = tree.select_next_agent(iteration=5)
    assert picked_4.id == "root"


def test_get_child_reports() -> None:
    tree = AgentTree(id="agents:4", max_active_agents=4, max_total_agents=6, max_depth=3)
    tree.ensure_root(agent_id="root", objective="root")
    
    tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="obj 1", child_id="c1"),
        iteration=1,
    )
    tree.spawn_child(
        parent_agent_id="root",
        request=AgentSpawnRequest(objective="obj 2", child_id="c2"),
        iteration=1,
    )
    
    # Neither are closed or have findings
    assert len(tree.get_child_reports("root")) == 0
    
    # Close c1 with findings
    c1_node = tree.nodes["c1"]
    c1_node.findings_summary = "c1 findings"
    c1_node.findings_confidence = 0.9
    tree.mark_closed(agent_id="c1", status=AgentNodeStatus.COMPLETED, iteration=2)
    
    reports = tree.get_child_reports("root")
    assert len(reports) == 1
    assert reports[0].id == "c1"
    assert reports[0].findings_summary == "c1 findings"
    
    # Close c2 without findings
    tree.mark_closed(agent_id="c2", status=AgentNodeStatus.COMPLETED, iteration=3)
    assert len(tree.get_child_reports("root")) == 1  # c2 has no findings yet
    
    # Add findings to c2
    c2_node = tree.nodes["c2"]
    c2_node.findings_summary = "c2 findings"
    
    reports = tree.get_child_reports("root")
    assert len(reports) == 2
