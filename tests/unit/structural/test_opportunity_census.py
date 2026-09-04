from __future__ import annotations

import pytest

from stream.structural.opportunity_census import (
    OpportunityCensusError,
    _classify_opportunity,
    _fixed_bitwidth,
    load_gate2b_contract,
    run_opportunity_census,
)


def test_gate2b_contract_pins_accepted_gate2a_artifact_and_claim_boundary():
    contract = load_gate2b_contract()

    assert contract["source_gate"]["sha256"] == (
        "sha256:d2473721f8e9d4796db5db7797ddf7c0099e82d4788ab4d081fe9b6a218aa145"
    )
    assert contract["source_gate"]["required_workloads"] == [
        "swiglu",
        "fsrcnn",
        "resnet18",
        "attention_head",
    ]
    assert "factor_coupling" in contract["excluded_claims"]
    assert "physical_traffic_volume" in contract["excluded_claims"]


@pytest.mark.parametrize(("type_name", "width"), [("i8", 8), ("bf16", 16), ("f32", 32)])
def test_fixed_bitwidth_is_derived_from_manifest_type(type_name: str, width: int):
    assert _fixed_bitwidth(type_name) == width


def test_non_fixed_bitwidth_type_is_rejected():
    with pytest.raises(OpportunityCensusError, match="unsupported fixed-bitwidth"):
        _fixed_bitwidth("index")


def test_classification_distinguishes_no_opportunity_placement_only_and_path_enabled():
    assert (
        _classify_opportunity(
            {
                "nondegenerate_tensor_count": 0,
                "path_nondegenerate_count": 0,
                "workloads_with_nondegenerate_tensor": 0,
                "workload_count": 1,
            },
        )[0]
        == "FAIL"
    )
    assert _classify_opportunity(
        {
            "nondegenerate_tensor_count": 1,
            "path_nondegenerate_count": 0,
            "workloads_with_nondegenerate_tensor": 1,
            "workload_count": 1,
        },
    )[:2] == ("NARROW", "PLACEMENT_ONLY")
    assert _classify_opportunity(
        {
            "nondegenerate_tensor_count": 1,
            "path_nondegenerate_count": 1,
            "workloads_with_nondegenerate_tensor": 1,
            "workload_count": 1,
        },
    )[:2] == ("PASS", "PLACEMENT_AND_PATH")


def test_gate2b_census_quantifies_frozen_real_workload_opportunity(tmp_path):
    report = run_opportunity_census(tmp_path / "report.json")

    assert report["verdict"] == "NARROW"
    assert report["opportunity_class"] == "PLACEMENT_ONLY"
    assert report["limitations"] == ["NO_INDEPENDENT_PATH_CHOICE"]
    assert report["summary"]["eligible_tensor_count"] == 65
    assert report["summary"]["nondegenerate_tensor_count"] == 17
    assert report["summary"]["placement_nondegenerate_count"] == 17
    assert report["summary"]["path_nondegenerate_count"] == 0
    assert report["summary"]["controlled_logical_bits_ratio"] == pytest.approx(0.023066978439384225)
    assert report["summary"]["largest_workload_assignment_space"] == {
        "workload": "fsrcnn",
        "size": "390625",
    }
    assert "naive_assignment_space" not in report["summary"]
    assert {
        workload_id: result["opportunity"]["naive_assignment_space"]
        for workload_id, result in report["workloads"].items()
    } == {
        "swiglu": "5",
        "fsrcnn": "390625",
        "resnet18": "15625",
        "attention_head": "25",
    }
