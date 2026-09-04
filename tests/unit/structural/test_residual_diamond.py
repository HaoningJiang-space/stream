from stream.structural.direct_cost import direct_cost
from stream.structural.exhaustive import exhaustive_minimize
from tests.unit.structural._cases import six_cases


def test_residual_diamond_has_feasible_structural_assignment():
    case = six_cases()["residual_diamond"]
    result = exhaustive_minimize(case.graph, lambda assignment: direct_cost(assignment, case.direct_events))

    assert result.optimum == 0
    assert result.feasible > 0
