from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PermissionLevel(str, Enum):
    READ = "read"
    WRITE = "write"
    SYSTEM = "system"


class ToolPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    permission_level: PermissionLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    requires_confirmation: bool = False
    max_invocations: int = Field(default=10, ge=1)
    allowed_phases: tuple[str, ...] = Field(default_factory=tuple)


class ToolPolicyStore:
    def __init__(self) -> None:
        self._policies: dict[str, ToolPolicy] = {}

    def register(self, tool_name: str, policy: ToolPolicy) -> None:
        self._policies[tool_name] = policy

    def get(self, tool_name: str) -> ToolPolicy:
        try:
            return self._policies[tool_name]
        except KeyError as exc:
            raise KeyError(f"No policy registered for tool '{tool_name}'") from exc


class ToolPolicyEnforcer:
    """Deterministic policy enforcement over invocation history."""

    def __init__(self, policy_store: ToolPolicyStore, max_risk_score: float) -> None:
        self._policy_store = policy_store
        self._max_risk_score = max_risk_score

    def validate(
        self,
        *,
        tool_name: str,
        current_phase: str,
        invocation_count: int,
        confirmation_granted: bool,
    ) -> ToolPolicy:
        policy = self._policy_store.get(tool_name)

        if policy.risk_score > self._max_risk_score:
            raise PermissionError(
                f"Tool '{tool_name}' blocked by risk policy ({policy.risk_score:.2f} > {self._max_risk_score:.2f})"
            )

        if policy.allowed_phases and current_phase not in policy.allowed_phases:
            if tool_name != "conclude":
                raise PermissionError(
                    f"Tool '{tool_name}' not allowed in phase '{current_phase}'"
                )

        if invocation_count >= policy.max_invocations:
            raise PermissionError(
                f"Tool '{tool_name}' exceeded max invocations ({policy.max_invocations})"
            )

        if policy.requires_confirmation and not confirmation_granted:
            raise PermissionError(f"Tool '{tool_name}' requires explicit confirmation")

        return policy
