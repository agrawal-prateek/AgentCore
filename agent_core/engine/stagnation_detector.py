from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field


class StagnationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    triggered: bool
    reasons: tuple[str, ...] = Field(default_factory=tuple)


@dataclass
class StagnationSignal:
    new_evidence_discovered: bool
    tool_name: str | None
    tool_args: dict[str, object] | None
    top_hypothesis_confidence: float
    branch_depth: int


class StagnationDetector:
    """Tracks drift/stagnation indicators over a bounded rolling window."""

    def __init__(self, *, threshold: int, max_depth: int, repeated_tool_window: int) -> None:
        self._threshold = threshold
        self._max_depth = max_depth
        self._recent_tool_fingerprints: deque[str] = deque(maxlen=repeated_tool_window)
        self._recent_confidences: deque[float] = deque(maxlen=threshold)
        self._no_new_evidence_streak = 0

    def evaluate(self, signal: StagnationSignal) -> StagnationReport:
        reasons: list[str] = []

        if signal.new_evidence_discovered:
            self._no_new_evidence_streak = 0
        else:
            self._no_new_evidence_streak += 1
        if self._no_new_evidence_streak >= self._threshold:
            reasons.append("no-new-evidence")

        if signal.tool_name is not None:
            fingerprint = self._fingerprint_tool_call(signal.tool_name, signal.tool_args or {})
            self._recent_tool_fingerprints.append(fingerprint)
            if (
                len(self._recent_tool_fingerprints) == self._recent_tool_fingerprints.maxlen
                and len(set(self._recent_tool_fingerprints)) == 1
            ):
                reasons.append("repeated-tool-call")

        self._recent_confidences.append(round(signal.top_hypothesis_confidence, 4))
        if (
            len(self._recent_confidences) == self._recent_confidences.maxlen
            and max(self._recent_confidences) - min(self._recent_confidences) <= 0.01
        ):
            reasons.append("unchanged-hypothesis-confidence")

        if signal.branch_depth > self._max_depth:
            reasons.append("excessive-branch-depth")

        return StagnationReport(triggered=bool(reasons), reasons=tuple(reasons))

    @staticmethod
    def _fingerprint_tool_call(tool_name: str, args: dict[str, object]) -> str:
        flattened = ",".join(f"{k}={args[k]}" for k in sorted(args))
        return sha256(f"{tool_name}|{flattened}".encode("utf-8")).hexdigest()
