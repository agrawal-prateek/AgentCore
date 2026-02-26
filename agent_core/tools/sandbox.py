from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agent_core.config.agent_config import SafetyMode
from agent_core.tools.tool_policy import PermissionLevel, ToolPolicy


class SandboxContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    read_only_mode: bool


class ToolSandbox:
    """Sandbox gate before tool execution."""

    def __init__(self, *, safety_mode: SafetyMode) -> None:
        self._safety_mode = safety_mode

    def assert_allowed(self, policy: ToolPolicy) -> SandboxContext:
        read_only_mode = self._safety_mode == SafetyMode.READ_ONLY

        if read_only_mode and policy.permission_level != PermissionLevel.READ:
            raise PermissionError("Non-read tool blocked in read-only safety mode")

        if policy.permission_level == PermissionLevel.SYSTEM and self._safety_mode != SafetyMode.STANDARD:
            raise PermissionError("System-level tool blocked by current safety mode")

        return SandboxContext(read_only_mode=read_only_mode)
