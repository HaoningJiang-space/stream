from __future__ import annotations

import json
from copy import deepcopy
from itertools import product

from stream.structural.pretiling_opportunity_census import (
    count_signature_compatibility,
    load_pretiling_opportunity_contract,
    pretiling_mapping_reference_matches,
    verify_pretiling_opportunity_provenance,
    workload_denominator_matches,
    write_pretiling_opportunity_provenance,
)


def _direct_count(domains):
    return sum(len({signature for signature in assignment if signature}) <= 1 for assignment in product(*domains))


def test_analytic_compatibility_matches_direct_multiset_enumeration():
    empty = ()
    split_2 = (("z0", 2),)
    split_4 = (("z0", 4),)
    left = [empty, empty, split_2, split_4]
    right = [empty, split_2, split_2, split_4]
    multiplicities = (
        {empty: 2, split_2: 1, split_4: 1},
        {empty: 1, split_2: 2, split_4: 1},
    )

    result = count_signature_compatibility(multiplicities)

    assert result["total_tuple_count"] == len(left) * len(right)
    assert result["compatible_tuple_count"] == _direct_count((left, right))
    assert result["all_empty_tuple_count"] == 2
    assert result["noncartesian"] is True


def test_compatibility_handles_missing_empty_state_and_unary_support():
    empty = ()
    split_2 = (("z0", 2),)
    split_4 = (("z0", 4),)
    left = {split_2: 2, split_4: 1}
    right = {empty: 1, split_2: 1}

    result = count_signature_compatibility((left, right))

    assert result["compatible_tuple_count"] == 5
    assert result["unary_projection_product"] == 6
    assert result["noncartesian"] is True


def test_all_empty_or_one_signature_class_is_cartesian():
    empty = ()
    split_2 = (("z0", 2),)

    result = count_signature_compatibility(({empty: 2, split_2: 1}, {empty: 1, split_2: 3}))

    assert result["compatible_tuple_count"] == result["total_tuple_count"]
    assert result["noncartesian"] is False


def test_empty_signature_without_compatible_remainder_has_no_unary_support():
    empty = ()
    split_2 = (("z0", 2),)
    split_4 = (("z0", 4),)

    result = count_signature_compatibility(({empty: 1}, {split_2: 1}, {split_4: 1}))

    assert result["compatible_tuple_count"] == 0
    assert result["unary_projection_product"] == 0
    assert result["noncartesian"] is False


def test_denominator_rejects_family_only_drift():
    denominator = {"workload": {"family": "residual", "group_count": 1, "operator_count": 2}}
    workloads = {
        "workload": {
            "family": "residual",
            "manifest": {
                "group_count": 1,
                "operator_count": 2,
                "denominator_matches_gate2a": True,
            },
        }
    }

    assert workload_denominator_matches(workloads, denominator) is True
    workloads["workload"]["family"] = "sequential"
    assert workload_denominator_matches(workloads, denominator) is False


def test_contract_freezes_pretiling_boundary_and_proxy_claim():
    contract = load_pretiling_opportunity_contract()

    assert contract["allowed_stage_trace"]["per_group"] == ["mapping_parser"]
    assert contract["execution"]["run_tiling_generation"] is False
    assert contract["execution"]["run_tta"] is False
    assert contract["signature_proxy"]["evidence_class"] == "PRETILING_SIGNATURE_PROXY"
    reference = contract["source_gates"]["gate2a"]["pretiling_reference"]
    assert reference["artifact"] == "artifacts/gate2f-a/gate2a-pretiling-reference-v4.json.gz.b64"
    assert reference["sha256"].startswith("sha256:")
    assert reference["instrument_commit"]
    assert reference["instrument_path"] == "scripts/capture_pretiling_mapping_reference.py"
    assert contract["correctness_criteria"]["gate2a_recorded_environment_match"] is True
    assert contract["correctness_criteria"]["gate2a_pretiling_reference_match"] is True
    assert "global_compatible_assignment_count" in contract["excluded_claims"]


def test_mapping_reference_requires_exact_group_mapping_and_generated_file_identity():
    reference_groups = [
        {
            "operator_ids": ["A", "B"],
            "workload_semantics": {"nodes": [{"id": "ComputationNode:A"}]},
            "mapping": {"nodes": {"A": {"resource_options": [[0]]}}},
            "mapping_file_sha256": "sha256:accepted",
        }
    ]
    groups = [
        {
            "operator_ids": ["ComputationNode:A", "ComputationNode:B"],
            "workload_semantics": {"nodes": [{"id": "ComputationNode:A"}]},
            "parsed_compute_mapping": {"nodes": {"A": {"resource_options": [[0]]}}},
            "mapping_file_sha256": "sha256:accepted",
        }
    ]

    assert pretiling_mapping_reference_matches(groups, reference_groups) is True
    groups[0]["mapping_file_sha256"] = "sha256:changed"
    assert pretiling_mapping_reference_matches(groups, reference_groups) is False
    groups[0]["mapping_file_sha256"] = "sha256:accepted"
    groups[0]["parsed_compute_mapping"]["nodes"]["A"]["resource_options"] = [[1]]
    assert pretiling_mapping_reference_matches(groups, reference_groups) is False
    groups[0]["parsed_compute_mapping"] = deepcopy(reference_groups[0]["mapping"])
    groups[0]["workload_semantics"] = {"nodes": [{"id": "ComputationNode:B"}]}
    assert pretiling_mapping_reference_matches(groups, reference_groups) is False
    groups[0]["workload_semantics"] = deepcopy(reference_groups[0]["workload_semantics"])
    groups[0]["operator_ids"] = ["ComputationNode:B", "ComputationNode:A"]
    assert pretiling_mapping_reference_matches(groups, reference_groups) is False


def test_provenance_bundle_detects_report_mutation(tmp_path):
    report_path = tmp_path / "report.json"
    report = {
        "run_status": "COMPLETED",
        "correctness_verdict": "PASS",
        "source": {"commit": "abc123"},
        "environment": {"host": "eex004", "python": "3.12"},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = write_pretiling_opportunity_provenance(
        report_path,
        report,
        ("python", "runner.py"),
        "completed\n",
        "",
    )

    assert verify_pretiling_opportunity_provenance(manifest_path) is True
    report_path.write_text("mutated\n", encoding="utf-8")
    assert verify_pretiling_opportunity_provenance(manifest_path) is False


def test_provenance_bundle_rejects_hash_consistent_invalid_report(tmp_path):
    report_path = tmp_path / "report.json"
    report = {
        "run_status": "INVALID",
        "correctness_verdict": "FAIL",
        "source": {"commit": "abc123"},
        "environment": {"host": "eex004", "python": "3.12"},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = write_pretiling_opportunity_provenance(
        report_path,
        report,
        ("python", "runner.py"),
        "invalid\n",
        "",
    )

    assert verify_pretiling_opportunity_provenance(manifest_path) is False


def test_provenance_bundle_rejects_inconsistent_host(tmp_path):
    report_path = tmp_path / "report.json"
    report = {
        "run_status": "COMPLETED",
        "correctness_verdict": "PASS",
        "source": {"commit": "abc123"},
        "environment": {"host": "eex004", "python": "3.12"},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = write_pretiling_opportunity_provenance(
        report_path,
        report,
        ("python", "runner.py"),
        "completed\n",
        "",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["host"] = "different-host"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_pretiling_opportunity_provenance(manifest_path) is False
