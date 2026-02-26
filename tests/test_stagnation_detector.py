from __future__ import annotations

from agent_core.engine.stagnation_detector import StagnationDetector, StagnationSignal


def test_stagnation_detector_triggers_on_repeated_patterns() -> None:
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


def test_stagnation_detector_triggers_on_excessive_depth() -> None:
    detector = StagnationDetector(threshold=4, max_depth=2, repeated_tool_window=3)
    report = detector.evaluate(
        StagnationSignal(
            new_evidence_discovered=True,
            tool_name=None,
            tool_args=None,
            top_hypothesis_confidence=0.3,
            branch_depth=3,
        )
    )

    assert report.triggered is True
    assert "excessive-branch-depth" in report.reasons
