from __future__ import annotations

import json

import pytest

from stream.structural.gate1a_cases import load_gate1a_accelerator
from stream.structural.operator_template_coupling import (
    _assignment_specs,
    _relation_not_cartesian,
    _run_assignment,
    load_operator_template_coupling_contract,
    run_operator_template_coupling,
    verify_operator_template_coupling_provenance,
    write_operator_template_coupling_provenance,
)


def test_gate1a_v4_denominator_is_the_frozen_pair_cross_product():
    contract = load_operator_template_coupling_contract()
    specs = _assignment_specs(contract)

    assert len(specs) == contract["expected_assignment_count"] == 98
    assert len(set(specs)) == len(specs)


def test_joint_split_restores_exact_executability(tmp_path):
    contract = load_operator_template_coupling_contract()
    result = _run_assignment(
        "fork_join",
        ("B", "C"),
        (4, 5),
        tmp_path,
        load_gate1a_accelerator(contract["hardware"]),
        contract,
    )

    assert result["paired_candidate_set_exact"] is True
    assert result["compiled_reference_outcome_exact"] is True
    assert result["compatibility_prediction_exact"] is True
    assert result["joint_recovery"] is True
    assert result["exact_executable"] is True
    assert result["silent_relaxations"] == 0


def test_mismatched_nonempty_splits_remain_explicitly_unsupported(tmp_path):
    contract = load_operator_template_coupling_contract()
    result = _run_assignment(
        "residual_diamond",
        ("A", "Add"),
        (4, 6),
        tmp_path,
        load_gate1a_accelerator(contract["hardware"]),
        contract,
    )

    assert result["compiled_reference_outcome_exact"] is True
    assert result["compatibility_prediction_exact"] is True
    assert result["compiled_outcome"]["reason"] == "SHARED_INPUT_TILING_INCOMPATIBLE"
    assert result["exact_executable"] is False


def test_joint_baseline_round_trip_is_compared_before_downstream_mutation(tmp_path):
    contract = load_operator_template_coupling_contract()
    result = _run_assignment(
        "fork_join",
        ("B", "C"),
        (6, 6),
        tmp_path,
        load_gate1a_accelerator(contract["hardware"]),
        contract,
    )

    assert result["baseline_round_trip_exact"] is True
    assert result["exact_executable"] is True


def test_forbidden_cross_witness_proves_relation_is_not_cartesian():
    coupled_pairs = {"fork_join": ("B", "C")}
    results = [
        {"assignment_id": "fork_join:B=4,C=4", "exact_executable": True},
        {"assignment_id": "fork_join:B=6,C=6", "exact_executable": True},
        {"assignment_id": "fork_join:B=4,C=6", "exact_executable": False},
        {"assignment_id": "fork_join:B=6,C=4", "exact_executable": False},
    ]

    assert _relation_not_cartesian(results, coupled_pairs) is True


def test_gate_rejects_worker_override_that_bypasses_frozen_replay(tmp_path):
    with pytest.raises(ValueError, match="requires exactly 8"):
        run_operator_template_coupling(tmp_path / "report.json", max_workers=1)


def test_runner_writes_hashed_provenance_sidecar(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text('{"verdict":"PASS"}\n', encoding="utf-8")
    report = {
        "source": {"commit": "frozen-commit"},
        "environment": {"python": "3.12", "packages": {"ortools": "9.15"}},
    }

    manifest_path = write_operator_template_coupling_provenance(
        report_path,
        report,
        ("python", "runner.py"),
        "PASS\n",
        "",
    )

    manifest = json.loads((tmp_path / "report.json.run.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == "frozen-commit"
    assert manifest["invocation"] == ["python", "runner.py"]
    assert len(manifest["report"]["sha256"]) == 64
    assert len(manifest["stdout"]["sha256"]) == 64
    assert len(manifest["stderr"]["sha256"]) == 64
    assert verify_operator_template_coupling_provenance(manifest_path) is True

    (tmp_path / "report.json.stdout.log").write_text("tampered\n", encoding="utf-8")
    assert verify_operator_template_coupling_provenance(manifest_path) is False
