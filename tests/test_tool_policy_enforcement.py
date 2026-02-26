from __future__ import annotations

import pytest

from agent_core.tools.tool_policy import PermissionLevel, ToolPolicy, ToolPolicyEnforcer, ToolPolicyStore


def test_tool_policy_blocks_phase_mismatch() -> None:
    store = ToolPolicyStore()
    store.register(
        "read_logs",
        ToolPolicy(
            permission_level=PermissionLevel.READ,
            risk_score=0.2,
            allowed_phases=("discover",),
        ),
    )

    enforcer = ToolPolicyEnforcer(policy_store=store, max_risk_score=0.9)

    with pytest.raises(PermissionError, match="not allowed in phase"):
        enforcer.validate(
            tool_name="read_logs",
            current_phase="validate",
            invocation_count=0,
            confirmation_granted=False,
        )


def test_tool_policy_blocks_risk_and_invocation_and_confirmation() -> None:
    store = ToolPolicyStore()
    store.register(
        "write_tool",
        ToolPolicy(
            permission_level=PermissionLevel.WRITE,
            risk_score=0.95,
            requires_confirmation=True,
            max_invocations=1,
            allowed_phases=("discover",),
        ),
    )

    enforcer = ToolPolicyEnforcer(policy_store=store, max_risk_score=0.7)

    with pytest.raises(PermissionError, match="risk policy"):
        enforcer.validate(
            tool_name="write_tool",
            current_phase="discover",
            invocation_count=0,
            confirmation_granted=True,
        )

    store.register(
        "safe_write",
        ToolPolicy(
            permission_level=PermissionLevel.WRITE,
            risk_score=0.4,
            requires_confirmation=True,
            max_invocations=1,
            allowed_phases=("discover",),
        ),
    )

    with pytest.raises(PermissionError, match="requires explicit confirmation"):
        enforcer.validate(
            tool_name="safe_write",
            current_phase="discover",
            invocation_count=0,
            confirmation_granted=False,
        )

    with pytest.raises(PermissionError, match="exceeded max invocations"):
        enforcer.validate(
            tool_name="safe_write",
            current_phase="discover",
            invocation_count=1,
            confirmation_granted=True,
        )


def test_tool_policy_allows_valid_call() -> None:
    store = ToolPolicyStore()
    policy = ToolPolicy(
        permission_level=PermissionLevel.READ,
        risk_score=0.1,
        requires_confirmation=False,
        max_invocations=4,
        allowed_phases=("discover", "validate"),
    )
    store.register("read_tool", policy)

    selected = ToolPolicyEnforcer(policy_store=store, max_risk_score=0.8).validate(
        tool_name="read_tool",
        current_phase="discover",
        invocation_count=2,
        confirmation_granted=False,
    )

    assert selected == policy
