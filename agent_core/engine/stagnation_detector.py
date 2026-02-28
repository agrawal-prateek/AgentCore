from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field

from agent_core.engine.similarity_port import SimilarityPort


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

    def __init__(
        self,
        *,
        threshold: int,
        max_depth: int,
        repeated_tool_window: int,
        dominance_threshold: float = 0.7,
        near_duplicate_similarity: float = 0.8,
        similarity_port: SimilarityPort | None = None,
    ) -> None:
        self._threshold = threshold
        self._max_depth = max_depth
        self._dominance_threshold = dominance_threshold
        self._near_duplicate_similarity = near_duplicate_similarity
        self._similarity_port = similarity_port
        self._recent_tool_fingerprints: deque[str] = deque(maxlen=repeated_tool_window)
        self._recent_confidences: deque[float] = deque(maxlen=threshold)
        self._recent_tool_names: deque[str] = deque(maxlen=repeated_tool_window)
        self._last_tool_query: dict[str, str] = {}  # tool_name -> last query text
        self._near_duplicate_streak = 0
        self._no_new_evidence_streak = 0

    async def evaluate_async(self, signal: StagnationSignal) -> StagnationReport:
        """Async evaluation — uses embedding similarity when port is available."""
        reasons = self._evaluate_sync_checks(signal)
        await self._check_near_duplicate(signal, reasons)
        return StagnationReport(triggered=bool(reasons), reasons=tuple(reasons))

    def evaluate(self, signal: StagnationSignal) -> StagnationReport:
        """Synchronous evaluation — skips embedding-based similarity."""
        reasons = self._evaluate_sync_checks(signal)
        return StagnationReport(triggered=bool(reasons), reasons=tuple(reasons))

    def _evaluate_sync_checks(self, signal: StagnationSignal) -> list[str]:
        """Core checks that don't require async."""
        reasons: list[str] = []

        # 1. No-new-evidence streak
        if signal.new_evidence_discovered:
            self._no_new_evidence_streak = 0
        else:
            self._no_new_evidence_streak += 1
        if self._no_new_evidence_streak >= self._threshold:
            reasons.append("no-new-evidence")

        # 2. Exact repeated tool call (existing)
        if signal.tool_name is not None:
            fingerprint = self._fingerprint_tool_call(signal.tool_name, signal.tool_args or {})
            self._recent_tool_fingerprints.append(fingerprint)
            if (
                len(self._recent_tool_fingerprints) == self._recent_tool_fingerprints.maxlen
                and len(set(self._recent_tool_fingerprints)) == 1
            ):
                reasons.append("repeated-tool-call")

            # 3. Tool-name dominance (NEW)
            self._recent_tool_names.append(signal.tool_name)
            if len(self._recent_tool_names) == self._recent_tool_names.maxlen:
                name_counts: dict[str, int] = {}
                for name in self._recent_tool_names:
                    name_counts[name] = name_counts.get(name, 0) + 1
                max_count = max(name_counts.values())
                dominance_ratio = max_count / len(self._recent_tool_names)
                if dominance_ratio >= self._dominance_threshold:
                    reasons.append("tool-name-dominance")

        # 4. Unchanged hypothesis confidence (existing)
        self._recent_confidences.append(round(signal.top_hypothesis_confidence, 4))
        if (
            len(self._recent_confidences) == self._recent_confidences.maxlen
            and max(self._recent_confidences) - min(self._recent_confidences) <= 0.01
        ):
            reasons.append("unchanged-hypothesis-confidence")

        # 5. Excessive branch depth (existing)
        if signal.branch_depth > self._max_depth:
            reasons.append("excessive-branch-depth")

        return reasons

    async def _check_near_duplicate(self, signal: StagnationSignal, reasons: list[str]) -> None:
        """Embedding-based near-duplicate detection."""
        if self._similarity_port is None or signal.tool_name is None:
            return

        query_text = self._extract_query_text(signal.tool_args or {})
        if not query_text:
            return

        last_query = self._last_tool_query.get(signal.tool_name)
        self._last_tool_query[signal.tool_name] = query_text

        if last_query is None or last_query == query_text:
            return

        try:
            score = await self._similarity_port.similarity(last_query, query_text)
        except Exception:
            return

        if score >= self._near_duplicate_similarity:
            self._near_duplicate_streak += 1
        else:
            self._near_duplicate_streak = 0

        if self._near_duplicate_streak >= 2:
            reasons.append("near-duplicate-tool-call")

    @staticmethod
    def _extract_query_text(args: dict[str, object]) -> str:
        """Extract the main query/search text from tool arguments."""
        for key in ("query", "q", "search", "pattern", "sql", "query_dsl"):
            val = args.get(key)
            if val is not None:
                return str(val)
        return ""

    @staticmethod
    def _fingerprint_tool_call(tool_name: str, args: dict[str, object]) -> str:
        flattened = ",".join(f"{k}={args[k]}" for k in sorted(args))
        return sha256(f"{tool_name}|{flattened}".encode("utf-8")).hexdigest()
