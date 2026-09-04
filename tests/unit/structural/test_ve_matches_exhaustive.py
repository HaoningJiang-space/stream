import pytest

from stream.structural.canonicalize import canonical_assignment
from stream.structural.direct_cost import direct_cost
from stream.structural.elimination import variable_elimination
from stream.structural.exhaustive import exhaustive_minimize
from stream.structural.micro_dags import six_cases


@pytest.mark.parametrize("case_name", tuple(six_cases()))
def test_variable_elimination_matches_independent_exhaustive_oracle(case_name):
    case = six_cases()[case_name]
    exhaustive = exhaustive_minimize(case.graph, lambda assignment: direct_cost(assignment, case.direct_events))
    eliminated = variable_elimination(case.graph)

    assert eliminated.optimum == exhaustive.optimum
    optimum_classes = {canonical_assignment(assignment) for assignment in exhaustive.assignments}
    assert canonical_assignment(eliminated.assignment) in optimum_classes
