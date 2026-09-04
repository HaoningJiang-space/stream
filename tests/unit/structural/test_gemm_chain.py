from stream.structural.direct_cost import direct_cost
from stream.structural.exhaustive import exhaustive_minimize
from stream.structural.micro_dags import six_cases


def test_gemm_chain_forced_layout_mismatch_charges_retile():
    case = six_cases()["gemm_chain"]
    result = exhaustive_minimize(case.graph, lambda assignment: direct_cost(assignment, case.direct_events))

    for assignment in result.assignments:
        events = case.direct_events(assignment)
        assert events is not None
        assert sum(event.kind == "retile" for event in events) == 1
