from __future__ import annotations

from agent_core.engine.termination_engine import TerminationEngine
from agent_core.state.hypothesis import Hypothesis


def test_termination_on_confidence_threshold(core_fixtures) -> None:
    fixtures = core_fixtures
    fixtures.memory.hypotheses["h1"].confidence_score = 0.95

    decision = TerminationEngine().evaluate(
        state=fixtures.state,
        memory=fixtures.memory,
        planner_termination_flag=False,
        stagnation_counter=0,
        risk_boundary_crossed=False,
    )

    assert decision.should_terminate is True
    assert decision.reason == "confidence-threshold-reached"


def test_termination_on_iteration_limit(core_fixtures) -> None:
    fixtures = core_fixtures
    fixtures.state.iteration_count = fixtures.config.max_iterations

    decision = TerminationEngine().evaluate(
        state=fixtures.state,
        memory=fixtures.memory,
        planner_termination_flag=False,
        stagnation_counter=0,
        risk_boundary_crossed=False,
    )

    assert decision.should_terminate is True
    assert decision.reason == "iteration-limit-reached"


def test_termination_on_stagnation(core_fixtures) -> None:
    fixtures = core_fixtures

    decision = TerminationEngine().evaluate(
        state=fixtures.state,
        memory=fixtures.memory,
        planner_termination_flag=False,
        stagnation_counter=fixtures.config.stagnation_threshold,
        risk_boundary_crossed=False,
    )

    assert decision.should_terminate is True
    assert decision.reason == "stagnation-threshold-exceeded"


def test_termination_on_risk(core_fixtures) -> None:
    fixtures = core_fixtures

    decision = TerminationEngine().evaluate(
        state=fixtures.state,
        memory=fixtures.memory,
        planner_termination_flag=False,
        stagnation_counter=0,
        risk_boundary_crossed=True,
    )

    assert decision.should_terminate is True
    assert decision.reason == "risk-boundary-crossed"


def test_termination_on_planner_flag(core_fixtures) -> None:
    fixtures = core_fixtures

    decision = TerminationEngine().evaluate(
        state=fixtures.state,
        memory=fixtures.memory,
        planner_termination_flag=True,
        stagnation_counter=0,
        risk_boundary_crossed=False,
    )

    assert decision.should_terminate is True
    assert decision.reason == "planner-signaled-termination"
