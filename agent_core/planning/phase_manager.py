from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_core.profiles.profile_interface import ProfileInterface


class PhaseTransitionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    changed: bool
    new_phase: str
    reason: str = Field(default="")
    disallowed_reason: str | None = Field(default=None)


class PhaseManager:
    """Profile-governed phase transitions."""

    def __init__(self, profile: ProfileInterface) -> None:
        self._profile = profile

    def validate_phase(self, phase: str) -> None:
        if phase not in self._profile.phases():
            raise ValueError(f"Unknown phase '{phase}'")

    def can_use_tool(self, phase: str, tool_name: str) -> bool:
        if tool_name == "conclude":
            return True
        phase_def = self._profile.phases().get(phase)
        if phase_def is None:
            return False
        return tool_name in phase_def.allowed_tools

    def transition(self, current_phase: str, requested_phase: str | None) -> PhaseTransitionResult:
        if requested_phase is None or requested_phase == current_phase:
            return PhaseTransitionResult(changed=False, new_phase=current_phase, reason="no-change")

        current = self._profile.phases().get(current_phase)
        if current is None:
            raise ValueError(f"Current phase '{current_phase}' not in profile")

        if requested_phase not in current.allowed_next_phases:
            disallowed = (
                f"Phase transition {current_phase}->{requested_phase} is not allowed. "
                f"From {current_phase} you may only transition to: "
                f"{', '.join(current.allowed_next_phases) or 'none (terminal phase)'}. "
                "Phase transitions are one-way and irreversible."
            )
            return PhaseTransitionResult(
                changed=False,
                new_phase=current_phase,
                reason=f"disallowed-transition:{current_phase}->{requested_phase}",
                disallowed_reason=disallowed,
            )

        return PhaseTransitionResult(changed=True, new_phase=requested_phase, reason="allowed")
