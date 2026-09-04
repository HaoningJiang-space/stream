import pytest

from stream.datatypes import LayerDim
from stream.execution_boundary import ExecutionEvent, ForbiddenExecutionError, audit_execution
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import TransferAndTensorAllocator
from stream.structural.elimination import variable_elimination
from stream.structural.exhaustive import exhaustive_minimize
from stream.structural.real_workload_lifting import (
    LiftingError,
    LiftingReason,
    _classify_error,
    _criteria_manifest,
    _next_preparation_stage,
    _require_complete_trace,
    _source_manifest,
    _source_run_manifest,
    _validate_semantic_exclusions,
    _validate_transfer_tiling_domains,
    _version_at_least,
    load_gate2a_contract,
)


def test_gate2a_contract_freezes_four_topology_families():
    contract = load_gate2a_contract()

    assert contract == {
        "contract": "Gate 2A Real-Workload Production Lifting Validity",
        "version": "gate2a-v1",
        "selection_policy": {
            "frozen_before_repairs": True,
            "families": ["transformer_ffn", "sequential_cnn", "residual_cnn", "attention_fork_join"],
        },
        "hardware": "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml",
        "workloads": [
            {
                "id": "swiglu",
                "family": "transformer_ffn",
                "path": "stream/inputs/examples/workload/swiglu_1024_4096_14336.onnx",
            },
            {
                "id": "fsrcnn",
                "family": "sequential_cnn",
                "path": "stream/inputs/examples/workload/fsrcnn.onnx",
            },
            {
                "id": "resnet18",
                "family": "residual_cnn",
                "path": "stream/inputs/examples/workload/resnet18.onnx",
            },
            {
                "id": "attention_head",
                "family": "attention_fork_join",
                "path": "stream/inputs/testing/workload/attention_head.onnx",
            },
        ],
        "repeat_count": 2,
        "preparation": {
            "mapping_generation": "generic_production",
            "fusion_cut_points": "derived_by_generic_mapping",
            "timeslot_policy": "resource_aware_production",
            "cost_lut": "unit_cost_prepare_only",
            "run_tta": False,
            "run_structural_search": False,
        },
        "environment": {"python": ">=3.12", "ortools": ">=9.15"},
        "semantic_exclusions": [
            {
                "operator_types": ["Conv", "Gemm", "MatmulSoftmax"],
                "input_indices": [2],
                "reason": "UNMODELED_ADDITIVE_OPERAND",
            },
            {
                "operator_types": ["Reshape", "Squeeze", "Unsqueeze"],
                "input_indices": [1],
                "reason": "SHAPE_METADATA_OPERAND",
            },
            {
                "operator_types": ["Slice"],
                "input_indices": [1, 2, 3, 4],
                "reason": "INDEX_METADATA_OPERAND",
            },
            {
                "operator_types": ["Gather"],
                "input_indices": [1],
                "reason": "INDEX_METADATA_OPERAND",
            },
            {
                "operator_types": ["LayerNormalization"],
                "input_indices": [1, 2],
                "reason": "UNMODELED_NORMALIZATION_PARAMETER",
            },
        ],
        "pass_criteria": {
            "lifting_success": 1.0,
            "deterministic_preparation": True,
            "provenance_coverage": 1.0,
            "empty_placement_domains": 0,
            "empty_path_domains": 0,
            "silent_fallbacks": 0,
            "undeclared_semantic_exclusions": 0,
            "forbidden_execution_events": 0,
        },
    }


def test_lifting_errors_have_stable_semantic_reason_codes():
    assert _classify_error(AssertionError("InEdge tensor bias must have been inferred")) is (
        LiftingReason.CONSTANT_TENSOR_INFERENCE_FAILURE
    )
    assert _classify_error(AssertionError("Multiple different inter-core tilings")) is (
        LiftingReason.TRANSFER_DOMAIN_INCOMPATIBLE
    )
    assert _classify_error(RuntimeError("Input tensor x has no producer")) is (
        LiftingReason.TENSOR_IDENTITY_INCONSISTENT
    )


def test_preparation_trace_identifies_first_unfinished_stage():
    assert _next_preparation_stage([]) == "transfer_graph"
    assert _next_preparation_stage(["transfer_graph", "fusion_splits", "mapping"]) == "cost_lut"
    assert (
        _next_preparation_stage(
            [
                "transfer_graph",
                "fusion_splits",
                "mapping",
                "cost_lut",
                "ssis",
                "iterations",
                "multiplicities",
                "timeslots",
            ]
        )
        == "prepare_problem"
    )


def test_environment_version_comparison_enforces_frozen_minimums():
    assert _version_at_least("3.12.0", "3.12")
    assert not _version_at_least("3.11.9", "3.12")
    assert _version_at_least("9.15.6755", "9.15")
    assert not _version_at_least("9.14.6206", "9.15")
    assert not _version_at_least("9.15.0rc1", "9.15")
    assert not _version_at_least("9.15.dev1", "9.15")
    assert not _version_at_least("unparseable", "9.15")
    assert not _version_at_least(None, "9.15")


def test_incomplete_or_reordered_preparation_trace_is_rejected():
    with pytest.raises(LiftingError, match="incomplete or reordered trace"):
        _require_complete_trace("prepare", ["transfer_graph", "mapping"], ("transfer_graph", "mapping", "ssis"))


def test_source_manifest_fails_closed_when_git_status_fails(monkeypatch, tmp_path):
    def git_checked(*arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return True, str(tmp_path)
        if arguments == ("rev-parse", "HEAD"):
            return True, "expected"
        if arguments == ("status", "--porcelain"):
            return False, ""
        raise AssertionError(arguments)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("stream.structural.real_workload_lifting._git_checked", git_checked)
    monkeypatch.setattr("stream.structural.real_workload_lifting._source_snapshot_digest", lambda: "snapshot")

    source = _source_manifest("expected")

    assert source["identified"] is False
    assert source["git_checks"] == {"root": True, "head": True, "status": False}


def test_source_run_manifest_requires_stable_checkout_and_external_output(tmp_path):
    source = {
        "commit": "expected",
        "identified": True,
        "git_checkout": True,
        "git_root": str(tmp_path),
        "git_checks": {"root": True, "head": True, "status": True},
        "head": "expected",
        "expected_commit_matches_head": True,
        "clean": True,
        "dirty_paths": [],
        "snapshot_digest": "snapshot",
    }

    outside = _source_run_manifest(source, dict(source), tmp_path.parent / "report.json")
    inside = _source_run_manifest(source, dict(source), tmp_path / "report.json")
    changed = _source_run_manifest(source, {**source, "head": "changed"}, tmp_path.parent / "report.json")

    assert outside["identified"] is True
    assert outside["stable_during_run"] is True
    assert inside["identified"] is False
    assert inside["output_outside_checkout"] is False
    assert changed["identified"] is False
    assert changed["stable_during_run"] is False


def test_undeclared_semantic_exclusion_is_rejected():
    exclusion = {
        "node": "conv",
        "operator_type": "Conv",
        "input_index": 2,
        "tensor": "bias",
        "shape": [16],
        "source": "initializer",
        "reason": "UNKNOWN_REASON",
    }

    with pytest.raises(LiftingError, match="unregistered or unshaped semantic exclusion"):
        _validate_semantic_exclusions([exclusion], load_gate2a_contract())


def _construct_tta():
    object.__new__(TransferAndTensorAllocator).__init__(None, None, None, 0, None, None, None, None)


def _solve_tta():
    object.__new__(TransferAndTensorAllocator).solve()


@pytest.mark.parametrize(
    ("event", "operation"),
    [
        (
            ExecutionEvent.TTA_CONSTRUCT,
            _construct_tta,
        ),
        (ExecutionEvent.TTA_SOLVE, _solve_tta),
        (ExecutionEvent.STRUCTURAL_EXHAUSTIVE, lambda: exhaustive_minimize(None, None)),
        (ExecutionEvent.STRUCTURAL_VARIABLE_ELIMINATION, lambda: variable_elimination(None)),
    ],
)
def test_prepare_only_boundary_rejects_every_forbidden_entrypoint(event, operation):
    with audit_execution(forbidden=frozenset({event})) as audit:
        with pytest.raises(ForbiddenExecutionError, match=event.value):
            operation()

    assert audit.manifest()[event.value] == 1


def test_criteria_manifest_requires_environment_and_every_structural_audit():
    contract = load_gate2a_contract()
    group = {
        "group_trace": ["mapping_parser", "kernel_state", "tiling_generation"],
        "preparation_trace": [
            "transfer_graph",
            "fusion_splits",
            "mapping",
            "cost_lut",
            "ssis",
            "iterations",
            "multiplicities",
            "timeslots",
        ],
        "mapping_fallbacks": [],
        "domains": {
            "provenance_coverage": 1.0,
            "tensors": [{"placement_count": 1}],
            "transfers": [{"path_count": 1}],
        },
    }
    result = {
        "valid": True,
        "deterministic": True,
        "manifest": {
            "groups": [group],
            "semantic_exclusions_audited": True,
            "execution_boundary": {event.value: 0 for event in ExecutionEvent},
        },
    }
    results = {workload["id"]: result for workload in contract["workloads"]}

    criteria = _criteria_manifest(contract, {"identified": True}, {"compatible": True}, results)

    assert all(criteria.values())
    assert not _criteria_manifest(contract, {"identified": True}, {"compatible": False}, results)[
        "environment_compatible"
    ]


def test_transfer_domain_audit_rejects_nondivisible_partition_width():
    dimension = LayerDim(position=0, prefix="z")
    transfer = type("Transfer", (), {"name": "transfer"})()
    node_mapping = type(
        "NodeMapping",
        (),
        {"inter_core_tiling": (((dimension, 3),),), "memory_allocation": ((0, 1, 2, 3),)},
    )()
    scheduler = type(
        "Scheduler",
        (),
        {
            "ssw": type(
                "Workload",
                (),
                {
                    "get_transfer_nodes": lambda _self: (transfer,),
                    "get_dims": lambda _self, _transfer: (dimension,),
                },
            )(),
            "mapping": type("Mapping", (), {"get": lambda _self, _transfer: node_mapping})(),
        },
    )()

    with pytest.raises(LiftingError, match="does not divide allocation width"):
        _validate_transfer_tiling_domains(scheduler)
