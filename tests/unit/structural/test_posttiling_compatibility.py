from __future__ import annotations

import json

from stream.cost_model.steady_state_scheduler import TransferLineage
from stream.structural.posttiling_compatibility import (
    _lineage_matches,
    _summary,
    load_posttiling_compatibility_contract,
    verify_posttiling_compatibility_provenance,
    write_posttiling_compatibility_provenance,
)


def test_contract_freezes_posttiling_relation_and_prepare_only_boundary():
    contract = load_posttiling_compatibility_contract()

    assert contract["version"] == "gate2f-b-v1"
    assert contract["relation"]["post"].startswith("production transfer-mapping acceptance")
    assert contract["execution"]["visualize_tiled_workload"] is False
    assert contract["execution"]["run_tta"] is False
    assert contract["execution"]["run_structural_search"] is False
    assert contract["faithfulness_criteria"] == {"false_positive_count": 0, "false_negative_count": 0}


def test_lineage_match_requires_tensor_producer_consumers_and_operand_indices():
    factor = {
        "tensor": "shared",
        "producer": "InEdge:input",
        "consumers": ["ComputationNode:A", "ComputationNode:B"],
        "consumer_operand_indices": [[0], [1]],
    }
    lineage = TransferLineage(
        "shared",
        "InEdge:input",
        ("ComputationNode:B", "ComputationNode:A"),
        ((1,), (0,)),
    )

    assert _lineage_matches(lineage, factor) is True
    assert _lineage_matches(TransferLineage("shared", "InEdge:input", lineage.consumers, ((0,), (0,))), factor) is False


def test_summary_keeps_invalid_rows_out_of_the_confusion_matrix():
    factors = {
        "factor": {
            "valid": True,
            "deterministic": True,
            "spec": {"total_tuple_count": 5},
            "manifest": {
                "execution_boundary": {
                    "tta_construct": 0,
                    "tta_solve": 0,
                    "structural_exhaustive": 0,
                    "structural_variable_elimination": 0,
                },
                "rows": [
                    {
                        "status": "VALID",
                        "pre": True,
                        "post": True,
                        "literal_survival": True,
                        "lineage_witness": True,
                        "nonempty_post_domains": True,
                    },
                    {
                        "status": "VALID",
                        "pre": True,
                        "post": False,
                        "literal_survival": True,
                        "lineage_witness": True,
                        "nonempty_post_domains": True,
                    },
                    {
                        "status": "VALID",
                        "pre": False,
                        "post": True,
                        "literal_survival": True,
                        "lineage_witness": True,
                        "nonempty_post_domains": True,
                    },
                    {
                        "status": "VALID",
                        "pre": False,
                        "post": False,
                        "literal_survival": True,
                        "lineage_witness": True,
                        "nonempty_post_domains": True,
                    },
                    {"status": "INVALID", "pre": True, "post": None},
                ],
            },
        }
    }

    summary = _summary(factors)

    assert summary["valid_tuple_count"] == 4
    assert summary["invalid_tuple_count"] == 1
    assert summary["true_positive_count"] == 1
    assert summary["true_negative_count"] == 1
    assert summary["false_positive_count"] == 1
    assert summary["false_negative_count"] == 1


def test_provenance_rejects_mutated_report(tmp_path):
    report_path = tmp_path / "report.json"
    report = {
        "run_status": "COMPLETED",
        "correctness_verdict": "PASS",
        "source": {"commit": "source"},
        "environment": {"host": "eex004", "python": "3.12"},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path = write_posttiling_compatibility_provenance(
        report_path,
        report,
        ("python", "runner.py"),
        "completed\n",
        "",
    )

    assert verify_posttiling_compatibility_provenance(manifest_path) is True
    report_path.write_text("mutated\n", encoding="utf-8")
    assert verify_posttiling_compatibility_provenance(manifest_path) is False
