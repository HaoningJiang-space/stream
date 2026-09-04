from stream.structural.direct_cost import assert_factor_accounting
from stream.structural.micro_dags import six_cases


def test_fork_join_factor_accounting_for_all_assignments():
    case = six_cases()["fork_join"]

    for assignment in case.graph.assignments():
        assert_factor_accounting(case.graph, assignment, case.direct_events)
