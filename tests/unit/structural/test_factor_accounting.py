import pytest

from stream.structural.direct_cost import assert_factor_accounting
from stream.structural.factors import FactorGraph, OwnedEventFactor, PhysicalEvent, Variable
from tests.unit.structural._cases import build_case


def test_every_assignment_has_identical_direct_and_factor_events():
    case = build_case((("t_a", "A", ("B", "C", "D")),))

    for assignment in case.graph.assignments():
        assert_factor_accounting(case.graph, assignment, case.direct_events)


def test_shared_materialization_is_owned_once_for_fanout():
    case = build_case((("t_a", "A", ("B", "C", "D")),))
    assignment = next(
        assignment
        for assignment in case.graph.assignments()
        if assignment["q:t_a"].materialization.value == "full"
        and assignment["q:t_a"].fragments[0].layout == assignment["A"].output_layout
        and assignment["d:t_a"].kind == "shared"
    )
    events = case.graph.factorized_events(assignment)

    assert events is not None
    assert [event.key for event in events].count("t_a:materialize") == 1
    assert sum(event.kind == "consume" for event in events) == 3


def test_duplicate_event_ownership_is_rejected():
    variable = Variable("x", (0,))
    one = OwnedEventFactor("one", ("x",), {(0,): (PhysicalEvent("move", "move", 1, "one"),)})
    two = OwnedEventFactor("two", ("x",), {(0,): (PhysicalEvent("move", "move", 1, "two"),)})
    graph = FactorGraph((variable,), (one, two))

    with pytest.raises(ValueError, match="charged more than once"):
        graph.factorized_cost({"x": 0})
