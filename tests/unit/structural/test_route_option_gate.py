from __future__ import annotations

import json

import pytest

from stream.structural.route_option_gate import (
    RouteOptionGateError,
    _comparison,
    _route_resource_signature,
    _validate_contract,
    load_gate2d_contract,
)


def _path(name: str, *, objective: float = 1.0) -> str:
    return json.dumps(
        {
            "sources": ["Core:0"],
            "targets": ["Core:1"],
            "total_hops_objective": objective,
            "links": [
                {
                    "sender": f"Core:{name}",
                    "receiver": "Core:1",
                    "bandwidth": 1,
                    "unit_energy_cost": 0,
                    "bidirectional": False,
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _workload_result(paths: list[str], *, placement: str = "cores:0", compatible_count: int | None = None) -> dict:
    count = compatible_count if compatible_count is not None else len(paths)
    return {
        "structural_invariant_hash": "sha256:structural",
        "domain_evidence": {
            "transfers": {
                "group-0:transfer": {
                    "path_keys": paths,
                    "route_resource_signatures": [_route_resource_signature(path) for path in paths],
                }
            },
            "staging_tensors": {
                "group-0:T": {
                    "adjacent_transfers": ["group-0:transfer"],
                    "placements": [placement],
                    "choices": {
                        placement: {
                            "assignment_count": count,
                            "compatible_path_counts": {"group-0:transfer": count},
                        }
                    },
                    "assignment_count": count,
                }
            },
        },
        "opportunity": {
            "path_nondegenerate_count": int(count > 1),
            "eligible_tensor_count": 1,
            "controlled_logical_bits_ratio": float(count > 1),
        },
    }


def _comparison_for(baseline: dict, expanded: dict) -> dict:
    contract = load_gate2d_contract()
    contract["source_gate"]["required_workloads"] = ["swiglu"]
    return _comparison(contract, {"1": {"swiglu": baseline}, "4": {"swiglu": expanded}})["swiglu"]


def test_gate2d_contract_separates_measured_and_static_invariants():
    contract = load_gate2d_contract()

    assert contract["route_plan_limits"] == [1, 4]
    assert contract["repeat_count"] == 2
    assert contract["expansion"]["measured_invariants"] == [
        "workload denominator",
        "operator tiling",
        "tensor and transfer identities",
        "placement candidate identities",
        "baseline semantic hashes",
        "route-insensitive structural manifest",
        "canonical baseline route resources remain available",
    ]
    assert contract["static_review"]["review_only_invariants"] == [
        "TTA objective",
        "timeslot algorithm",
        "constraint semantics",
    ]
    assert "latency_improvement" in contract["excluded_claims"]


def test_gate2d_contract_rejects_unevaluated_criterion():
    contract = load_gate2d_contract()
    contract["pass_criteria"]["unknown"] = True

    with pytest.raises(RouteOptionGateError, match="pass criteria"):
        _validate_contract(contract)


def test_route_domain_comparison_proves_identity_inclusion_and_strict_growth():
    baseline = _workload_result([_path("a")])
    expanded = _workload_result([_path("a"), _path("b")])

    comparison = _comparison_for(baseline, expanded)

    assert comparison == {
        "same_tensor_variables": True,
        "same_transfer_variables": True,
        "placement_domains_identical": True,
        "route_resource_signatures_unique": True,
        "path_domains_included": True,
        "compatible_path_counts_monotone": True,
        "strict_path_domain_expansion": True,
        "structural_invariant_hash_match": True,
        "expanded_path_nondegenerate_tensor_ratio": 1.0,
        "expanded_controlled_logical_bits_ratio": 1.0,
    }


def test_route_domain_comparison_rejects_duplicate_route_resources():
    path = _path("a")
    comparison = _comparison_for(_workload_result([path]), _workload_result([path, path]))

    assert comparison["route_resource_signatures_unique"] is False


def test_route_resource_signature_ignores_objective_only_differences():
    assert _route_resource_signature(_path("a", objective=1.0)) == _route_resource_signature(
        _path("a", objective=99.0)
    )


def test_route_domain_comparison_rejects_replaced_baseline_paths():
    comparison = _comparison_for(
        _workload_result([_path("a")]),
        _workload_result([_path("b"), _path("c")]),
    )

    assert comparison["path_domains_included"] is False


def test_route_domain_comparison_checks_placement_identity_not_count():
    comparison = _comparison_for(
        _workload_result([_path("a")], placement="cores:0"),
        _workload_result([_path("a"), _path("b")], placement="cores:1"),
    )

    assert comparison["placement_domains_identical"] is False
