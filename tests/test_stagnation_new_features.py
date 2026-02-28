"""Tests for new stagnation detector features: tool-name dominance and near-duplicate detection."""

from __future__ import annotations

import pytest

from agent_core.engine.similarity_port import SimilarityPort
from agent_core.engine.stagnation_detector import StagnationDetector, StagnationSignal


class FakeSimilarityPort(SimilarityPort):
    """Fake similarity port that returns configurable scores."""

    def __init__(self, score: float = 0.95) -> None:
        self._score = score

    async def similarity(self, text_a: str, text_b: str) -> float:
        return self._score


def test_tool_name_dominance_triggers() -> None:
    detector = StagnationDetector(
        threshold=5,
        max_depth=4,
        repeated_tool_window=4,
        dominance_threshold=0.7,
    )

    # First 3 calls all same tool
    for _ in range(3):
        detector.evaluate(
            StagnationSignal(
                new_evidence_discovered=True,
                tool_name="semantic_code_search",
                tool_args={"query": f"different query {_}"},
                top_hypothesis_confidence=0.5,
                branch_depth=1,
            )
        )

    # 4th call makes dominance = 4/4 = 100% >= 70%
    report = detector.evaluate(
        StagnationSignal(
            new_evidence_discovered=True,
            tool_name="semantic_code_search",
            tool_args={"query": "yet another query"},
            top_hypothesis_confidence=0.55,
            branch_depth=1,
        )
    )

    assert report.triggered is True
    assert "tool-name-dominance" in report.reasons


def test_tool_name_dominance_does_not_trigger_when_diverse() -> None:
    detector = StagnationDetector(
        threshold=5,
        max_depth=4,
        repeated_tool_window=4,
        dominance_threshold=0.7,
    )

    tools = ["semantic_code_search", "grep_code", "view_file", "search_logs"]
    for tool in tools:
        report = detector.evaluate(
            StagnationSignal(
                new_evidence_discovered=True,
                tool_name=tool,
                tool_args={"query": "test"},
                top_hypothesis_confidence=0.5,
                branch_depth=1,
            )
        )

    assert "tool-name-dominance" not in report.reasons


@pytest.mark.asyncio
async def test_near_duplicate_detection_with_high_similarity() -> None:
    similarity_port = FakeSimilarityPort(score=0.95)
    detector = StagnationDetector(
        threshold=5,
        max_depth=4,
        repeated_tool_window=4,
        near_duplicate_similarity=0.8,
        similarity_port=similarity_port,
    )

    # Need 2+ consecutive near-duplicates to trigger
    for i in range(3):
        report = await detector.evaluate_async(
            StagnationSignal(
                new_evidence_discovered=True,
                tool_name="semantic_code_search",
                tool_args={"query": f"error handling in payment service v{i}"},
                top_hypothesis_confidence=0.5,
                branch_depth=1,
            )
        )

    assert report.triggered is True
    assert "near-duplicate-tool-call" in report.reasons


@pytest.mark.asyncio
async def test_near_duplicate_not_triggered_without_port() -> None:
    detector = StagnationDetector(
        threshold=5,
        max_depth=4,
        repeated_tool_window=4,
    )

    for i in range(3):
        report = await detector.evaluate_async(
            StagnationSignal(
                new_evidence_discovered=True,
                tool_name="semantic_code_search",
                tool_args={"query": f"error handling v{i}"},
                top_hypothesis_confidence=0.5,
                branch_depth=1,
            )
        )

    assert "near-duplicate-tool-call" not in report.reasons


@pytest.mark.asyncio
async def test_near_duplicate_not_triggered_with_low_similarity() -> None:
    similarity_port = FakeSimilarityPort(score=0.3)
    detector = StagnationDetector(
        threshold=5,
        max_depth=4,
        repeated_tool_window=4,
        near_duplicate_similarity=0.8,
        similarity_port=similarity_port,
    )

    for i in range(3):
        report = await detector.evaluate_async(
            StagnationSignal(
                new_evidence_discovered=True,
                tool_name="semantic_code_search",
                tool_args={"query": f"completely different {i}"},
                top_hypothesis_confidence=0.5,
                branch_depth=1,
            )
        )

    assert "near-duplicate-tool-call" not in report.reasons


def test_backward_compat_existing_stagnation_still_triggers() -> None:
    """Verify the original stagnation patterns still work."""
    detector = StagnationDetector(threshold=3, max_depth=4, repeated_tool_window=3)

    report = None
    for _ in range(3):
        report = detector.evaluate(
            StagnationSignal(
                new_evidence_discovered=False,
                tool_name="search",
                tool_args={"q": "same"},
                top_hypothesis_confidence=0.42,
                branch_depth=1,
            )
        )

    assert report is not None
    assert report.triggered is True
    assert "no-new-evidence" in report.reasons
    assert "repeated-tool-call" in report.reasons
    assert "unchanged-hypothesis-confidence" in report.reasons
