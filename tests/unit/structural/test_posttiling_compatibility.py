from __future__ import annotations

import json
import subprocess
from time import perf_counter
from types import SimpleNamespace

from stream.cost_model.steady_state_scheduler import TensorRelevantTilingDecision, TransferLineage
from stream.datatypes import LayerDim
from stream.structural.posttiling_compatibility import (
    _cartesian_tuple_at,
    _factor_explains_ssis_rejection,
    _isolated_factor_attempt,
    _lineage_matches,
    _merge_factor_shards,
    _summary,
    _translated_fused_group_projection,
    load_posttiling_compatibility_contract,
    verify_posttiling_compatibility_provenance,
    write_posttiling_compatibility_provenance,
)
from stream.workload.utils import SpatialUnrollingExtentError


def test_fused_group_projection_uses_current_workload_dimension_coordinates(monkeypatch):
    dimensions = (SimpleNamespace(position=1), SimpleNamespace(position=2))
    translated_positions = {1: 2, 2: 1}
    mapping = SimpleNamespace(
        fused_groups=(
            SimpleNamespace(
                name="group",
                layers=("A", "B"),
                intra_core_tiling=((dimensions[0], 32), (dimensions[1], 112)),
            ),
        )
    )
    monkeypatch.setattr(
        "stream.structural.posttiling_compatibility.get_equivalent_dimension",
        lambda _source, _current, dimension: SimpleNamespace(position=translated_positions[dimension.position]),
    )

    assert _translated_fused_group_projection(object(), mapping, object()) == [
        {
            "name": "group",
            "layers": ["A", "B"],
            "intra_core_tiling": [{"position": 2, "factor": 32}, {"position": 1, "factor": 112}],
        }
    ]


def test_contract_freezes_posttiling_relation_and_prepare_only_boundary():
    contract = load_posttiling_compatibility_contract()

    assert contract["version"] == "gate2f-b-v1"
    assert contract["relation"]["post"].startswith("production acceptance through SSIS construction")
    assert contract["execution"]["visualize_tiled_workload"] is False
    assert contract["execution"]["run_tta"] is False
    assert contract["execution"]["run_structural_search"] is False
    assert contract["faithfulness_criteria"] == {"false_positive_count": 0, "false_negative_count": 0}


def test_ssis_rejection_requires_a_matching_completed_factor_decision():
    dimension = LayerDim(position=4, prefix="z")
    lineage = TransferLineage(
        "shared",
        "InEdge:shared",
        ("ComputationNode:A", "ComputationNode:B"),
        ((0,), (0,)),
    )
    decision = TensorRelevantTilingDecision(
        transfer="Transfer(shared)",
        lineage=lineage,
        projections=(("A", ()), ("B", ((dimension, 4),))),
        selected_reference=None,
        role="completed_transfer_mapping",
        accepted=True,
        result_resource_option_count=4,
        result_memory_option_count=1,
    )

    assert _factor_explains_ssis_rejection([decision], SpatialUnrollingExtentError(dimension, 6, 4, ("B",))) is True
    assert (
        _factor_explains_ssis_rejection([decision], SpatialUnrollingExtentError(dimension, 6, 4, ("unrelated",)))
        is False
    )
    assert _factor_explains_ssis_rejection([decision], SpatialUnrollingExtentError(dimension, 6, 2, ("B",))) is False


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
                        "post_relation_witness": True,
                        "literal_survival": True,
                        "lineage_witness": True,
                        "nonempty_post_domains": True,
                        "ssis_semantics": True,
                    },
                    {
                        "status": "VALID",
                        "pre": True,
                        "post": False,
                        "post_relation_witness": True,
                        "literal_survival": True,
                        "lineage_witness": True,
                        "nonempty_post_domains": True,
                        "ssis_semantics": True,
                    },
                    {
                        "status": "VALID",
                        "pre": False,
                        "post": True,
                        "post_relation_witness": True,
                        "literal_survival": True,
                        "lineage_witness": True,
                        "nonempty_post_domains": True,
                        "ssis_semantics": True,
                    },
                    {
                        "status": "VALID",
                        "pre": False,
                        "post": False,
                        "post_relation_witness": True,
                        "literal_survival": True,
                        "lineage_witness": True,
                        "nonempty_post_domains": True,
                        "ssis_semantics": True,
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
    assert summary["accepted_tuple_count"] == 2
    assert summary["rejected_tuple_count"] == 2


def test_worker_timeout_is_a_terminal_environment_failure(monkeypatch, tmp_path):
    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("worker", 1)

    monkeypatch.setattr(subprocess, "run", time_out)
    result = _isolated_factor_attempt(
        {"workload_id": "w", "group": 0, "factor": 0, "total_tuple_count": 1},
        tmp_path,
        0,
        {},
        perf_counter() + 1,
    )

    assert result["status"] == "ENVIRONMENT_FAILURE"
    assert "deadline" in result["detail"]


def test_factor_shards_merge_in_global_ordinal_order():
    spec = {"workload_id": "w", "group": 0, "factor": 0, "total_tuple_count": 3}
    boundary = {
        "tta_construct": 0,
        "tta_solve": 0,
        "structural_exhaustive": 0,
        "structural_variable_elimination": 0,
    }

    def shard(start, stop):
        return {
            "status": "VALID",
            "hash_seed": "3000",
            "wall_seconds": 1.0,
            "manifest": {
                "spec": spec,
                "frontend_trace": ["mapping_parser"],
                "mapping_parser_reference_match": True,
                "domain_hashes_match": True,
                "factor_manifest_match": True,
                "tuple_start": start,
                "tuple_stop": stop,
                "tuple_key_lines": [f"key-{ordinal}" for ordinal in range(start, stop)],
                "witnesses": {},
                "rows": [{"ordinal": ordinal} for ordinal in range(start, stop)],
                "execution_boundary": boundary,
            },
        }

    result = _merge_factor_shards(spec, 0, {2: shard(2, 3), 0: shard(0, 2)}, 2)

    assert result["status"] == "VALID"
    assert [row["ordinal"] for row in result["manifest"]["rows"]] == [0, 1, 2]
    assert "tuple_key_lines" not in result["manifest"]


def test_factor_shards_reject_seed_and_key_cardinality_drift():
    spec = {"workload_id": "w", "group": 0, "factor": 0, "total_tuple_count": 1}
    manifest = {
        "spec": spec,
        "frontend_trace": [],
        "mapping_parser_reference_match": True,
        "domain_hashes_match": True,
        "factor_manifest_match": True,
        "tuple_start": 0,
        "tuple_stop": 1,
        "tuple_key_lines": [],
        "witnesses": {},
        "rows": [{"ordinal": 0}],
        "execution_boundary": {
            "tta_construct": 0,
            "tta_solve": 0,
            "structural_exhaustive": 0,
            "structural_variable_elimination": 0,
        },
    }

    wrong_seed = _merge_factor_shards(
        spec, 0, {0: {"status": "VALID", "hash_seed": "wrong", "wall_seconds": 1.0, "manifest": manifest}}, 1
    )
    bad_cardinality = _merge_factor_shards(
        spec, 0, {0: {"status": "VALID", "hash_seed": "3000", "wall_seconds": 1.0, "manifest": manifest}}, 1
    )

    assert wrong_seed["status"] == "CORRECTNESS_FAILURE"
    assert "hash-seed" in wrong_seed["detail"]
    assert bad_cardinality["status"] == "CORRECTNESS_FAILURE"
    assert "cardinality" in bad_cardinality["detail"]


def test_mixed_radix_cartesian_lookup_matches_product_order():
    domains = (("a", "b"), (1, 2, 3))

    assert [_cartesian_tuple_at(domains, ordinal) for ordinal in range(6)] == [
        ("a", 1),
        ("a", 2),
        ("a", 3),
        ("b", 1),
        ("b", 2),
        ("b", 3),
    ]


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
