from __future__ import annotations

from stream.structural.coupling_census import _scope_metrics, load_gate2c_contract, run_coupling_census


def test_gate2c_contract_pins_model_and_excludes_tetra_separability_claim():
    contract = load_gate2c_contract()

    assert contract["source_gate"]["sha256"] == (
        "sha256:49a7a09edcdbae137c72ed5ffdfc2bf3b021d9809ab71d2f36ba1584043bf971"
    )
    assert contract["audited_structural_model"]["sha256"] == (
        "sha256:a30439a300647d0303e73662afe9b44b622fc9f9b9bc3ad2c66159d22ac58082"
    )
    assert "TETRA_feasible_set_separability" in contract["excluded_claims"]


def test_scope_metrics_detect_actual_non_unary_interaction():
    metrics = _scope_metrics(["a", "b", "c"], [("a",), ("a", "b"), ("c",)])

    assert metrics["non_unary_factor_count"] == 1
    assert metrics["primal_graph_edge_count"] == 1
    assert metrics["largest_coupled_component"] == 2
    assert metrics["interaction_variable_ratio"] == 2 / 3
    assert metrics["induced_width"] is None
    assert metrics["inference_regime"] == "COUPLED_NOT_SOLVED"


def test_gate2c_census_proves_current_executable_proxy_is_separable(tmp_path):
    report = run_coupling_census(tmp_path / "report.json")

    assert report["verdict"] == "PASS"
    assert report["inference_regime"] == "SEPARABLE"
    assert report["summary"] == {
        "workload_count": 4,
        "variable_count": 17,
        "factor_count": 17,
        "non_unary_factor_count": 0,
        "primal_graph_edge_count": 0,
        "largest_coupled_component": 1,
        "interaction_variable_ratio": 0.0,
        "induced_width": 0,
        "independent_argmin_evaluations": 85,
    }
    assert {
        workload_id: result["independent_argmin_evaluations"] for workload_id, result in report["workloads"].items()
    } == {"swiglu": 5, "fsrcnn": 40, "resnet18": 30, "attention_head": 10}
