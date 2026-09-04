"""Gate 2A production-path, prepare-only lifting census for real workloads."""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.cost_model.core_cost import CoreCostEntry
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.execution_boundary import ExecutionEvent, ForbiddenExecutionError, audit_execution
from stream.hardware.architecture.core import Core
from stream.opt.allocation.constraint_optimization.tensor_restriction import (
    tensor_placement_key,
    transfer_plan_key,
)
from stream.opt.solver import ConstraintSelection
from stream.stages.context import StageContext
from stream.stages.generation.generic_mapping_generation import GenericMappingGenerationStage
from stream.stages.generation.kernel_state import KernelStateStage
from stream.stages.generation.normalization_expansion import ExpandNormalizationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.mapping_parser import MappingParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.stages.stage import LeafStage, MainStage, StageCallable
from stream.structural.stream_contract import canonical_mapping_manifest
from stream.workload.iterator_type import is_state_operand
from stream.workload.node import ComputationNode, HasInputs, HasIterationSpace, HasOutputs, InEdge, TransferNode
from stream.workload.tensor import Tensor

_MIN_STAGING_TRANSFERS = 2
_MAX_AUTOMATIC_WORKERS = 8
_GATE2A_TRANSFER_PLAN_LIMIT = 1
_WORKER_ARGUMENT_COUNT = 3
_FRONTEND_STAGES = (
    "accelerator_parser",
    "onnx_parser",
    "normalization_expansion",
    "generic_mapping",
)
_GROUP_STAGES = ("mapping_parser", "kernel_state", "tiling_generation")
_PREPARATION_STAGES = (
    "transfer_graph",
    "fusion_splits",
    "mapping",
    "cost_lut",
    "ssis",
    "iterations",
    "multiplicities",
    "timeslots",
)


class LiftingReason(str, Enum):
    """Stable Gate 2A outcome codes."""

    LIFT_OK = "LIFT_OK"
    FRONTEND_INVALID = "FRONTEND_INVALID"
    AFFINE_DOMAIN_INVALID = "AFFINE_DOMAIN_INVALID"
    CONSTANT_TENSOR_INFERENCE_FAILURE = "CONSTANT_TENSOR_INFERENCE_FAILURE"
    TILING_INCONSISTENT = "TILING_INCONSISTENT"
    TRANSFER_DOMAIN_INCOMPATIBLE = "TRANSFER_DOMAIN_INCOMPATIBLE"
    SHARED_INPUT_DEMAND_UNREPRESENTABLE = "SHARED_INPUT_DEMAND_UNREPRESENTABLE"
    TENSOR_IDENTITY_INCONSISTENT = "TENSOR_IDENTITY_INCONSISTENT"
    EMPTY_PLACEMENT_DOMAIN = "EMPTY_PLACEMENT_DOMAIN"
    EMPTY_PATH_DOMAIN = "EMPTY_PATH_DOMAIN"
    MAPPING_FALLBACK = "MAPPING_FALLBACK"
    NONDETERMINISTIC_PREPARATION = "NONDETERMINISTIC_PREPARATION"
    UNDECLARED_SEMANTIC_EXCLUSION = "UNDECLARED_SEMANTIC_EXCLUSION"
    FORBIDDEN_EXECUTION = "FORBIDDEN_EXECUTION"


@dataclass(frozen=True, slots=True)
class LiftingError(RuntimeError):
    """A preparation failure with a stable stage and reason code."""

    stage: str
    reason: LiftingReason
    detail: str

    def __str__(self) -> str:
        return f"{self.stage}: {self.reason.value}: {self.detail}"


def load_gate2a_contract() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath("gate2a_contract.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _contract_digest() -> str:
    resource = files("stream.structural.contracts").joinpath("gate2a_contract.json")
    return "sha256:" + sha256(resource.read_bytes()).hexdigest()


def run_real_workload_lifting(
    output_path: str | Path,
    *,
    source_commit: str | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run the frozen census without constructing or solving TTA."""

    contract = load_gate2a_contract()
    workload_specs = contract["workloads"]
    repeat_count = int(contract["repeat_count"])
    attempt_count = len(workload_specs) * repeat_count
    worker_count = _resolve_worker_count(max_workers, attempt_count)
    destination = Path(output_path).resolve()
    source_before = _source_manifest(source_commit)
    environment = _environment_manifest(contract)
    hardware_path = Path(contract["hardware"])
    preparation_started = perf_counter()
    with TemporaryDirectory(prefix="stream-gate2a-") as temporary:
        workload_results = _run_preparation_attempts(
            workload_specs,
            hardware_path,
            Path(temporary),
            repeat_count,
            worker_count,
            _GATE2A_TRANSFER_PLAN_LIMIT,
        )
    preparation_wall_seconds = perf_counter() - preparation_started

    source_after = _source_manifest(source_commit)
    source = _source_run_manifest(source_before, source_after, destination)
    criteria = _criteria_manifest(contract, source, environment, workload_results)
    passed = all(criteria.values())
    payload = {
        "contract": contract,
        "contract_sha256": _contract_digest(),
        "verdict": "PASS" if passed else "NOT_RUN",
        "criteria": criteria,
        "source": source,
        "environment": environment,
        "execution": {
            "attempt_count": attempt_count,
            "max_workers": worker_count,
            "max_transfer_plans_per_endpoint": _GATE2A_TRANSFER_PLAN_LIMIT,
            "preparation_wall_seconds": round(preparation_wall_seconds, 6),
        },
        "hardware": {"path": str(hardware_path), "sha256": _file_digest(hardware_path)},
        "workloads": workload_results,
        "summary": _summary(workload_results),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(destination)
    return payload


def _resolve_worker_count(requested: int | None, attempt_count: int) -> int:
    if attempt_count < 1:
        raise ValueError("Gate 2A requires at least one preparation attempt")
    if requested is not None and requested < 1:
        raise ValueError("max_workers must be at least one")
    available_cpus = os.cpu_count() or 1
    limit = requested if requested is not None else _MAX_AUTOMATIC_WORKERS
    return min(attempt_count, available_cpus, limit)


def _run_preparation_attempts(
    workload_specs: list[dict[str, str]],
    hardware_path: Path,
    temporary_root: Path,
    repeat_count: int,
    max_workers: int,
    max_transfer_plans_per_endpoint: int,
) -> dict[str, Any]:
    attempts_by_workload: dict[str, list[dict[str, Any] | None]] = {
        workload_spec["id"]: [None] * repeat_count for workload_spec in workload_specs
    }
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gate2a") as executor:
        pending = {}
        for workload_spec in workload_specs:
            workload_id = workload_spec["id"]
            for repeat in range(repeat_count):
                future = executor.submit(
                    _run_isolated_attempt,
                    workload_spec,
                    hardware_path,
                    temporary_root / f"{workload_id}-{repeat}",
                    repeat,
                    max_transfer_plans_per_endpoint,
                )
                pending[future] = (workload_id, repeat)
        for future in as_completed(pending):
            workload_id, repeat = pending[future]
            attempts_by_workload[workload_id][repeat] = future.result()

    results = {}
    for workload_spec in workload_specs:
        workload_id = workload_spec["id"]
        ordered_attempts = attempts_by_workload[workload_id]
        if any(attempt is None for attempt in ordered_attempts):
            raise RuntimeError(f"missing preparation attempt for {workload_id}")
        results[workload_id] = _summarize_workload_preparation(
            workload_spec,
            [attempt for attempt in ordered_attempts if attempt is not None],
            repeat_count,
        )
    return results


def _summarize_workload_preparation(
    workload_spec: dict[str, str], attempts: list[dict[str, Any]], repeat_count: int
) -> dict[str, Any]:
    successful = [attempt for attempt in attempts if "manifest" in attempt]
    hashes = [_digest(attempt["manifest"]) for attempt in successful]
    deterministic = len(successful) == repeat_count and len(set(hashes)) == 1
    if successful and not deterministic:
        attempts.append(
            {
                "reason": LiftingReason.NONDETERMINISTIC_PREPARATION.value,
                "stage": "repeat_comparison",
                "detail": f"semantic hashes differ: {hashes}",
            }
        )
    canonical_manifest = successful[0]["manifest"] if deterministic else None
    if deterministic:
        for attempt, semantic_hash in zip(successful, hashes, strict=True):
            attempt["semantic_hash"] = semantic_hash
            del attempt["manifest"]
    return {
        "family": workload_spec["family"],
        "path": workload_spec["path"],
        "sha256": _file_digest(Path(workload_spec["path"])),
        "valid": deterministic,
        "deterministic": deterministic,
        "process_isolated": True,
        "semantic_hash": hashes[0] if deterministic else None,
        "manifest": canonical_manifest,
        "attempts": attempts,
    }


def _run_isolated_attempt(
    workload_spec: dict[str, str],
    hardware_path: Path,
    work_dir: Path,
    repeat: int,
    max_transfer_plans_per_endpoint: int,
) -> dict[str, Any]:
    """Prepare once in a fresh interpreter with a repeat-specific hash seed."""

    started = perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "worker-request.json"
    result_path = work_dir / "worker-result.json"
    request_path.write_text(
        json.dumps(
            {
                "workload_spec": workload_spec,
                "hardware_path": str(hardware_path),
                "work_dir": str(work_dir / "preparation"),
                "result_path": str(result_path),
                "max_transfer_plans_per_endpoint": max_transfer_plans_per_endpoint,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    hash_seed = str(1000 + repeat)
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = hash_seed
    completed = subprocess.run(
        (sys.executable, "-m", "stream.structural.real_workload_lifting", "--worker-request", str(request_path)),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0 or not result_path.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip() or "worker produced no result"
        return {
            "repeat": repeat,
            "hash_seed": hash_seed,
            "wall_seconds": round(perf_counter() - started, 6),
            "reason": LiftingReason.FRONTEND_INVALID.value,
            "stage": "worker_process",
            "detail": f"exit={completed.returncode}: {detail[-2000:]}",
        }
    attempt = json.loads(result_path.read_text(encoding="utf-8"))
    attempt.update(
        {
            "repeat": repeat,
            "hash_seed": hash_seed,
            "wall_seconds": round(perf_counter() - started, 6),
        }
    )
    return attempt


def _run_preparation_worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = Path(request["result_path"])
    try:
        with audit_execution(forbidden=frozenset(ExecutionEvent)) as execution_audit:
            manifest = _prepare_once(
                request["workload_spec"],
                Path(request["hardware_path"]),
                Path(request["work_dir"]),
                max_transfer_plans_per_endpoint=int(request["max_transfer_plans_per_endpoint"]),
            )
        manifest["execution_boundary"] = execution_audit.manifest()
        result = {
            "reason": LiftingReason.LIFT_OK.value,
            "manifest": manifest,
        }
    except ForbiddenExecutionError as failure:
        result = {
            "reason": LiftingReason.FORBIDDEN_EXECUTION.value,
            "stage": "execution_boundary",
            "detail": str(failure),
        }
    except LiftingError as failure:
        result = {"reason": failure.reason.value, "stage": failure.stage, "detail": failure.detail}
    except Exception as error:  # noqa: BLE001 - preserve unclassified production failures in the report
        result = {
            "reason": _classify_error(error).value,
            "stage": "unclassified",
            "detail": f"{type(error).__name__}: {error}",
        }
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


def _prepare_once(
    workload_spec: dict[str, str],
    hardware_path: Path,
    work_dir: Path,
    *,
    max_transfer_plans_per_endpoint: int = _GATE2A_TRANSFER_PLAN_LIMIT,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    context = StageContext.from_kwargs(
        accelerator=str(hardware_path),
        workload_path=workload_spec["path"],
        output_path=str(work_dir),
        intra_core_tiling=None,
        fusion_cut_points=None,
    )
    frontend_trace: list[str] = []
    for stage, label in (
        (AcceleratorParserStage, "accelerator_parser"),
        (ONNXModelParserStage, "onnx_parser"),
        (ExpandNormalizationStage, "normalization_expansion"),
        (GenericMappingGenerationStage, "generic_mapping"),
    ):
        try:
            context = _run_stage(stage, context)
            frontend_trace.append(label)
        except Exception as error:
            raise LiftingError(label, _classify_error(error), f"{type(error).__name__}: {error}") from error
    _require_complete_trace("frontend", frontend_trace, _FRONTEND_STAGES)

    onnx_model = context.require_value("onnx_model", "Gate2A")
    semantic_exclusions = _validate_semantic_exclusions(
        context.get("semantic_exclusions", ()),
        load_gate2a_contract(),
    )
    source_names = {
        "model_input": {item.name for item in onnx_model.graph.input},
        "constant": {item.name for item in onnx_model.graph.initializer},
    }
    sub_workloads = context.require_value("sub_workloads", "Gate2A")
    mapping_paths = context.require_value("group_mapping_paths", "Gate2A")
    if not sub_workloads:
        raise LiftingError("generic_mapping", LiftingReason.FRONTEND_INVALID, "no fusion groups were generated")
    if len(sub_workloads) != len(mapping_paths):
        raise LiftingError(
            "generic_mapping",
            LiftingReason.TENSOR_IDENTITY_INCONSISTENT,
            f"{len(sub_workloads)} groups but {len(mapping_paths)} mappings",
        )

    group_manifests = []
    for index, (sub_workload, mapping_path) in enumerate(zip(sub_workloads, mapping_paths, strict=True)):
        group_manifests.append(
            _prepare_group(
                index,
                sub_workload,
                mapping_path,
                context.get("accelerator"),
                source_names,
                work_dir,
                max_transfer_plans_per_endpoint,
            )
        )
    return {
        "frontend_trace": frontend_trace,
        "semantic_exclusions_audited": True,
        "semantic_exclusions": semantic_exclusions,
        "group_count": len(group_manifests),
        "groups": group_manifests,
    }


def _prepare_group(
    index: int,
    sub_workload,
    mapping_path: str,
    accelerator,
    source_names: dict[str, set[str]],
    work_dir: Path,
    max_transfer_plans_per_endpoint: int,
) -> dict[str, Any]:
    group_dir = work_dir / f"group_{index}"
    group_dir.mkdir(parents=True, exist_ok=True)
    context = StageContext.from_kwargs(
        accelerator=accelerator,
        workload=sub_workload,
        mapping_path=mapping_path,
        output_path=str(group_dir),
    )
    group_trace = []
    for stage, label in (
        (MappingParserStage, "mapping_parser"),
        (KernelStateStage, "kernel_state"),
        (TilingGenerationStage, "tiling_generation"),
    ):
        try:
            context = _run_stage(stage, context)
            group_trace.append(label)
        except Exception as error:
            raise LiftingError(label, _classify_error(error), f"{type(error).__name__}: {error}") from error
    _require_complete_trace("group_preparation", group_trace, _GROUP_STAGES)

    workload = context.get("workload")
    mapping = context.get("mapping")
    if not workload.get_computation_nodes():
        raise LiftingError("group_preparation", LiftingReason.FRONTEND_INVALID, "fusion group has no computation")
    _validate_affine_domains(workload)
    fallbacks = _mapping_fallbacks(workload, mapping)
    if fallbacks:
        raise LiftingError("mapping_audit", LiftingReason.MAPPING_FALLBACK, ", ".join(fallbacks))
    cost_lut = _unit_cost_lut(workload, mapping)
    scheduler = SteadyStateScheduler(
        workload,
        accelerator,
        mapping,
        context.get("fusion_splits"),
        cost_lut,
        backend="ORTOOLS_GSCIP",
        constraint_selection=ConstraintSelection(),
        total_mac_ops=context.get("total_mac_ops"),
        max_transfer_plans_per_endpoint=max_transfer_plans_per_endpoint,
    )
    preparation_trace: list[str] = []
    try:
        prepared = scheduler.prepare_problem(preparation_observer=preparation_trace.append)
    except Exception as error:
        stage = _next_preparation_stage(preparation_trace)
        raise LiftingError(stage, _classify_error(error), f"{type(error).__name__}: {error}") from error
    _require_complete_trace("scheduler_preparation", preparation_trace, _PREPARATION_STAGES)
    if not scheduler.ssw.get_computation_nodes() or not scheduler.ssw.get_transfer_nodes():
        raise LiftingError(
            "scheduler_preparation",
            LiftingReason.FRONTEND_INVALID,
            "prepared group must contain computation and transfer nodes",
        )
    _validate_transfer_tiling_domains(scheduler)
    domain_manifest = _option_domain_manifest(scheduler, source_names)
    return {
        "group": index,
        "group_trace": group_trace,
        "preparation_trace": preparation_trace,
        "input_workload": _workload_manifest(workload),
        "steady_state_workload": _workload_manifest(scheduler.ssw),
        "mapping": canonical_mapping_manifest(prepared.mapping),
        "timeslots": sorted((_entity_id(node), slot) for node, slot in prepared.timeslots.items()),
        "iterations": scheduler.iterations,
        "multiplicities": sorted(
            (_entity_id(node), multiplicity) for node, multiplicity in prepared.multiplicities.items()
        ),
        "ssis": _ssis_manifest(scheduler),
        "domains": domain_manifest,
        "mapping_fallbacks": [],
    }


def _run_stage(stage: StageCallable, context: StageContext) -> StageContext:
    results = MainStage([stage, LeafStage], context).run()
    if len(results) != 1:
        raise RuntimeError(f"{stage.__name__} returned {len(results)} contexts")
    return results[0]


def _validate_semantic_exclusions(
    exclusions: tuple[dict[str, Any], ...] | list[dict[str, Any]], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    """Require every supplied-but-unmodeled ONNX operand to match a frozen exclusion rule."""

    rules = contract["semantic_exclusions"]
    audited = []
    for exclusion in exclusions:
        declared = any(
            exclusion["operator_type"] in rule["operator_types"]
            and exclusion["input_index"] in rule["input_indices"]
            and exclusion["reason"] == rule["reason"]
            for rule in rules
        )
        if not declared or exclusion["shape"] is None:
            raise LiftingError(
                "semantic_exclusion_audit",
                LiftingReason.UNDECLARED_SEMANTIC_EXCLUSION,
                f"unregistered or unshaped semantic exclusion: {exclusion}",
            )
        audited.append(dict(exclusion))
    return sorted(
        audited,
        key=lambda item: (item["node"], item["input_index"], item["tensor"]),
    )


def _unit_cost_lut(workload, mapping) -> CoreCostLUT:
    """Create a deterministic placeholder; Gate 2A never reads performance costs."""

    cost_lut = CoreCostLUT(load=False)
    for node in workload.get_computation_nodes():
        node_mapping = mapping.get(node)
        if len(node_mapping.resource_allocation) != 1 or not node_mapping.resource_allocation[0]:
            raise LiftingError(
                "cost_lut",
                LiftingReason.EMPTY_PLACEMENT_DOMAIN,
                f"computation {node.name!r} has no unique allocation group",
            )
        for core in node_mapping.resource_allocation[0]:
            cost_lut.add_cost(node, core, CoreCostEntry(1, 1, 1, 1))
    return cost_lut


def _validate_affine_domains(workload) -> None:
    try:
        for node in workload.get_iteration_space_nodes():
            if len(node.operand_mapping) != len(node.tensors):
                raise ValueError(f"{node.name}: operand-map count differs from tensor count")
            if any(mapping.num_dims != node.num_dims for mapping in node.operand_mapping):
                raise ValueError(f"{node.name}: operand maps have inconsistent iteration ranks")
            for tensor, mapping in zip(node.tensors, node.operand_mapping, strict=True):
                if len(mapping.results) != len(tensor.shape) or any(size <= 0 for size in tensor.shape):
                    raise ValueError(f"{node.name}/{tensor.name}: invalid affine tensor domain")
    except Exception as error:
        raise LiftingError(
            "affine_domain_audit",
            LiftingReason.AFFINE_DOMAIN_INVALID,
            f"{type(error).__name__}: {error}",
        ) from error


def _mapping_fallbacks(workload, mapping) -> list[str]:
    fallbacks = []
    for node in workload.get_computation_nodes():
        for allocation in mapping.get(node).resource_allocation:
            for core in allocation:
                supported = core.operator_types is None or node.type in core.operator_types
                if not supported:
                    fallbacks.append(f"{node.name}->{core.id}")
    return sorted(set(fallbacks))


def _validate_transfer_tiling_domains(scheduler: SteadyStateScheduler) -> None:
    """Reject transfer mappings whose partition width cannot map onto their allocation."""

    for transfer in scheduler.ssw.get_transfer_nodes():
        node_mapping = scheduler.mapping.get(transfer)
        tilings = tuple(node_mapping.inter_core_tiling)
        allocations = tuple(node_mapping.memory_allocation)
        if len(tilings) != len(allocations):
            raise LiftingError(
                "transfer_domain_audit",
                LiftingReason.TRANSFER_DOMAIN_INCOMPATIBLE,
                f"{transfer.name}: {len(tilings)} tilings for {len(allocations)} allocations",
            )
        transfer_dimensions = set(scheduler.ssw.get_dims(transfer))
        for tiling, allocation in zip(tilings, allocations, strict=True):
            if any(dimension not in transfer_dimensions for dimension, _ in tiling):
                raise LiftingError(
                    "transfer_domain_audit",
                    LiftingReason.TRANSFER_DOMAIN_INCOMPATIBLE,
                    f"{transfer.name}: tiling contains a dimension outside the transfer domain",
                )
            partition_width = prod(factor for _, factor in tiling)
            if partition_width < 1 or len(allocation) % partition_width != 0:
                raise LiftingError(
                    "transfer_domain_audit",
                    LiftingReason.TRANSFER_DOMAIN_INCOMPATIBLE,
                    f"{transfer.name}: partition width {partition_width} does not divide allocation width "
                    f"{len(allocation)}",
                )


def _option_domain_manifest(scheduler: SteadyStateScheduler, source_names: dict[str, set[str]]) -> dict[str, Any]:
    if scheduler.ssw is None:
        raise LiftingError("option_domains", LiftingReason.FRONTEND_INVALID, "missing steady-state workload")
    tensors: dict[Tensor, tuple[Core, ...] | tuple[tuple[Core, ...], ...]] = {}
    producers: dict[Tensor, HasOutputs] = {}
    for node in scheduler.ssw.topological_sort():
        if not isinstance(node, HasOutputs):
            continue
        carried = (
            tuple(tensor for tensor in node.inputs if is_state_operand(node, tensor))
            if isinstance(node, HasIterationSpace)
            else ()
        )
        for tensor in (*node.outputs, *carried):
            if tensor in tensors:
                continue
            raw = _placement_options(scheduler, node)
            tensors[tensor] = raw
            producers[tensor] = node

    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        raise LiftingError(
            "option_domains",
            LiftingReason.TENSOR_IDENTITY_INCONSISTENT,
            f"non-unique tensor IDs: {duplicates}",
        )

    transfer_rows = []
    transfer_domains: dict[TransferNode, tuple[MulticastPathPlan, ...]] = {}
    for transfer in scheduler.ssw.get_transfer_nodes():
        options = tuple(scheduler.mapping.get(transfer).resource_allocation)
        if not options or not all(isinstance(option, MulticastPathPlan) for option in options):
            raise LiftingError(
                "option_domains", LiftingReason.EMPTY_PATH_DOMAIN, f"transfer {transfer.name!r} has no path domain"
            )
        transfer_domains[transfer] = options
        transfer_rows.append(
            {
                "id": transfer.name,
                "path_count": len(options),
                "paths": sorted(transfer_plan_key(option) for option in options),
            }
        )

    tensor_rows = []
    restrictable = []
    for tensor in sorted(tensors, key=lambda item: item.name):
        placements = _normalize_placements(tensors[tensor], tensor.name)
        producer = producers[tensor]
        adjacent = tuple(
            sorted(
                (transfer for transfer in scheduler.ssw.get_transfer_nodes() if tensor in transfer.tensors),
                key=lambda item: item.name,
            )
        )
        provenance = _tensor_provenance(scheduler, tensor, producer, source_names)
        row = {
            "id": tensor.name,
            "producer": _entity_id(producer),
            "consumers": sorted(
                _entity_id(node)
                for node in scheduler.ssw.nodes
                if isinstance(node, HasInputs) and tensor in node.inputs
            ),
            "provenance": provenance,
            "shape": list(tensor.shape),
            "placement_count": len(placements),
            "placements": sorted(tensor_placement_key(option) for option in placements),
            "adjacent_transfers": [transfer.name for transfer in adjacent],
        }
        tensor_rows.append(row)
        if isinstance(producer, TransferNode) and len(adjacent) >= _MIN_STAGING_TRANSFERS:
            choices = _compatible_staging_choices(tensor, placements, adjacent, transfer_domains)
            restrictable.append(
                {
                    "tensor": tensor.name,
                    "placement_count": len(placements),
                    "assignment_count": sum(choice["assignment_count"] for choice in choices),
                    "nondegenerate": sum(choice["assignment_count"] for choice in choices) > 1,
                    "choices": choices,
                }
            )
    if not tensor_rows:
        raise LiftingError("option_domains", LiftingReason.FRONTEND_INVALID, "prepared workload has no tensors")
    return {
        "tensor_count": len(tensor_rows),
        "transfer_count": len(transfer_rows),
        "provenance_coverage": sum(bool(row["provenance"]) for row in tensor_rows) / len(tensor_rows),
        "tensors": tensor_rows,
        "transfers": sorted(transfer_rows, key=lambda row: row["id"]),
        "staging_tensors": sorted(restrictable, key=lambda row: row["tensor"]),
    }


def _placement_options(scheduler: SteadyStateScheduler, producer: HasOutputs):
    if isinstance(producer, InEdge):
        return ((scheduler.accelerator.get_core(scheduler.accelerator.offchip_core_id),),)
    node_mapping = scheduler.mapping.get(producer)
    return node_mapping.memory_allocation if isinstance(producer, TransferNode) else node_mapping.resource_allocation


def _normalize_placements(raw, tensor_name: str) -> tuple[tuple[Core, ...], ...]:
    options = tuple(raw)
    if not options:
        raise LiftingError(
            "option_domains", LiftingReason.EMPTY_PLACEMENT_DOMAIN, f"tensor {tensor_name!r} has no placement domain"
        )
    if all(isinstance(item, Core) for item in options):
        options = (options,)
    normalized = tuple(tuple(option) for option in options)
    if any(not option or not all(isinstance(core, Core) for core in option) for option in normalized):
        raise LiftingError(
            "option_domains", LiftingReason.EMPTY_PLACEMENT_DOMAIN, f"tensor {tensor_name!r} has invalid placements"
        )
    return normalized


def _compatible_staging_choices(tensor, placements, adjacent, transfer_domains) -> list[dict[str, Any]]:
    choices = []
    for placement in placements:
        placement_key = tensor_placement_key(placement)
        compatible_counts = []
        for transfer in adjacent:
            count = sum(
                _path_endpoint_placement_key(path, transfer, tensor) == placement_key
                for path in transfer_domains[transfer]
            )
            if count == 0:
                raise LiftingError(
                    "option_domains",
                    LiftingReason.EMPTY_PATH_DOMAIN,
                    f"{transfer.name!r} has no path compatible with {tensor.name!r} at {placement_key}",
                )
            compatible_counts.append(count)
        assignment_count = 1
        for count in compatible_counts:
            assignment_count *= count
        choices.append(
            {
                "placement": placement_key,
                "compatible_path_counts": compatible_counts,
                "assignment_count": assignment_count,
            }
        )
    return choices


def _path_endpoint_placement_key(path: MulticastPathPlan, transfer: TransferNode, tensor: Tensor) -> str | None:
    groups = []
    if tensor in transfer.inputs:
        groups.append(tuple(path.sources))
    if tensor in transfer.outputs:
        groups.append(tuple(path.targets))
    keys = {tensor_placement_key(group) for group in groups}
    return next(iter(keys)) if len(keys) == 1 else None


def _tensor_provenance(
    scheduler: SteadyStateScheduler,
    tensor: Tensor,
    producer: HasOutputs,
    source_names: dict[str, set[str]],
) -> str:
    if isinstance(producer, ComputationNode) and tensor in producer.inputs and is_state_operand(producer, tensor):
        return "state"
    current_tensor = tensor
    current = producer
    visited = set()
    while isinstance(current, TransferNode):
        if current in visited or len(current.inputs) != 1:
            raise LiftingError(
                "provenance", LiftingReason.TENSOR_IDENTITY_INCONSISTENT, f"invalid transfer lineage for {tensor.name}"
            )
        visited.add(current)
        current_tensor = current.inputs[0]
        predecessors = tuple(scheduler.ssw.predecessors(current))
        if len(predecessors) != 1 or not isinstance(predecessors[0], HasOutputs):
            raise LiftingError(
                "provenance", LiftingReason.TENSOR_IDENTITY_INCONSISTENT, f"invalid producer for {tensor.name}"
            )
        current = predecessors[0]
    if isinstance(current, ComputationNode):
        return "activation"
    if isinstance(current, InEdge):
        if current_tensor.name in source_names["constant"] or current.name in source_names["constant"]:
            return "constant"
        if current_tensor.name in source_names["model_input"] or current.name in source_names["model_input"]:
            return "model_input"
        return "fused_intermediate"
    raise LiftingError("provenance", LiftingReason.TENSOR_IDENTITY_INCONSISTENT, f"unknown producer for {tensor.name}")


def _ssis_manifest(scheduler: SteadyStateScheduler) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "entity": _entity_id(entity),
                "variables": [
                    {
                        "dimension": str(variable.dimension),
                        "size": variable.size,
                        "effect": variable.effect.name,
                        "type": variable.type.name,
                        "reuse": variable.reuse.name,
                    }
                    for variable in ssis.variables
                ],
            }
            for entity, ssis in scheduler.ssis.items()
        ),
        key=lambda row: row["entity"],
    )


def _workload_manifest(workload) -> dict[str, Any]:
    """Canonical graph/domain view without recomputing the symbolic unique basis.

    The prepared SSIS manifest already records the resolved global ``z`` dimensions. This view
    retains the source affine maps, global loop positions, and inferred extents, which is enough
    to hash the input graph without repeating the expensive symbolic elimination.
    """

    dimension_sizes = workload.get_dimension_sizes()
    nodes = []
    for node in workload.dataflow_sort():
        row: dict[str, Any] = {"id": _entity_id(node)}
        if isinstance(node, HasInputs):
            row["inputs"] = [
                {"name": tensor.name, "shape": list(tensor.shape), "type": str(tensor.operand_type)}
                for tensor in node.inputs
            ]
        if isinstance(node, HasOutputs):
            row["outputs"] = [
                {"name": tensor.name, "shape": list(tensor.shape), "type": str(tensor.operand_type)}
                for tensor in node.outputs
            ]
        if isinstance(node, HasIterationSpace):
            positions = workload.global_idxs[node]
            row["dimensions"] = [
                {"global_position": position, "size": dimension_sizes[position]} for position in positions
            ]
            row["operand_maps"] = [str(mapping) for mapping in node.operand_mapping]
        if isinstance(node, ComputationNode):
            row["operation"] = str(node.type)
        if isinstance(node, TransferNode):
            row["transfer_type"] = node.transfer_type.name
        nodes.append(row)
    edges = []
    for source, target in workload.edges:
        shared = []
        if isinstance(source, HasOutputs) and isinstance(target, HasInputs):
            shared = [tensor.name for tensor in source.outputs if tensor in target.inputs]
        edges.append({"source": _entity_id(source), "target": _entity_id(target), "tensors": shared})
    return {
        "nodes": nodes,
        "edges": edges,
        "dimension_sizes": list(dimension_sizes),
    }


def _classify_error(error: Exception) -> LiftingReason:
    message = str(error)
    if "must have been inferred" in message:
        return LiftingReason.CONSTANT_TENSOR_INFERENCE_FAILURE
    if "Multiple different inter-core tilings" in message:
        return LiftingReason.TRANSFER_DOMAIN_INCOMPATIBLE
    if "incompatible tensor-relevant tilings" in message:
        return LiftingReason.SHARED_INPUT_DEMAND_UNREPRESENTABLE
    if "no producer" in message or "Expected exactly one source" in message:
        return LiftingReason.TENSOR_IDENTITY_INCONSISTENT
    if "tiling" in message.lower() or "unrolling" in message.lower():
        return LiftingReason.TILING_INCONSISTENT
    return LiftingReason.FRONTEND_INVALID


def _next_preparation_stage(completed: list[str]) -> str:
    return next((stage for stage in _PREPARATION_STAGES if stage not in completed), "prepare_problem")


def _require_complete_trace(stage: str, actual: list[str], expected: tuple[str, ...]) -> None:
    if tuple(actual) != expected:
        raise LiftingError(
            stage,
            LiftingReason.FRONTEND_INVALID,
            f"incomplete or reordered trace: expected {list(expected)}, got {actual}",
        )


def _source_manifest(expected_commit: str | None, *, executed_module_path: str | Path | None = None) -> dict[str, Any]:
    root_ok, root = _git_checked("rev-parse", "--show-toplevel")
    resolved_root = Path(root).resolve() if root_ok else None
    inside_git = resolved_root == Path.cwd().resolve()
    executed_module = Path(executed_module_path or __file__).resolve()
    module_within_checkout = resolved_root is not None and executed_module.is_relative_to(resolved_root)
    head_ok, head = _git_checked("rev-parse", "HEAD") if inside_git else (False, "")
    status_ok, status = _git_checked("status", "--porcelain") if inside_git else (False, "")
    commit = expected_commit or head
    commit_matches = not expected_commit or expected_commit == head
    return {
        "commit": commit,
        "identified": (
            inside_git
            and module_within_checkout
            and head_ok
            and status_ok
            and bool(head)
            and commit_matches
            and not status
        ),
        "git_checkout": inside_git,
        "git_root": root or None,
        "executed_module": str(executed_module),
        "module_within_checkout": module_within_checkout,
        "git_checks": {"root": root_ok, "head": head_ok, "status": status_ok},
        "head": head or None,
        "expected_commit_matches_head": commit_matches,
        "clean": not status if inside_git else None,
        "dirty_paths": status.splitlines(),
        "snapshot_digest": _source_snapshot_digest(),
    }


def _source_run_manifest(before: dict[str, Any], after: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Combine pre/post-run Git checks and require evidence output outside the checkout."""

    roots = {item["git_root"] for item in (before, after) if item["git_root"]}
    stable = (
        len(roots) == 1
        and before["head"] == after["head"]
        and before["snapshot_digest"] == after["snapshot_digest"]
        and before["dirty_paths"] == after["dirty_paths"]
    )
    root = Path(next(iter(roots))).resolve() if len(roots) == 1 else None
    output_outside_checkout = root is not None and not destination.is_relative_to(root)
    return {
        **after,
        "identified": bool(before["identified"] and after["identified"] and stable and output_outside_checkout),
        "stable_during_run": stable,
        "output_outside_checkout": output_outside_checkout,
        "pre_run": before,
        "post_run": after,
    }


def _source_snapshot_digest() -> str:
    roots = (Path("stream"), Path("scripts/run_real_workload_lifting.py"))
    entries = []
    for root in roots:
        candidates = root.rglob("*") if root.is_dir() else (root,)
        for path in candidates:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json"}:
                entries.append((str(path), _file_digest(path)))
    return _digest(sorted(entries))


def _environment_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    python_version = platform.python_version()
    packages = {name: _package_version(name) for name in ("onnx", "ortools", "zigzag-dse", "stream-dse")}
    requirements = contract["environment"]
    checks = {
        "python_minimum": _version_at_least(python_version, requirements["python"].removeprefix(">=")),
        "ortools_minimum": _version_at_least(packages["ortools"], requirements["ortools"].removeprefix(">=")),
    }
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_version": python_version,
        "packages": packages,
        "requirements": requirements,
        "checks": checks,
        "compatible": all(checks.values()),
    }


def _version_at_least(actual: str | None, minimum: str) -> bool:
    """Compare dotted release components without adding a runtime dependency."""

    if actual is None or re.fullmatch(r"\d+(?:\.\d+)*", actual) is None:
        return False

    def release(value: str) -> tuple[int, ...]:
        components = []
        for part in value.split("."):
            match = re.match(r"\d+", part)
            if match is None:
                break
            components.append(int(match.group()))
        return tuple(components)

    actual_release = release(actual)
    minimum_release = release(minimum)
    if not actual_release or not minimum_release:
        return False
    width = max(len(actual_release), len(minimum_release))
    return actual_release + (0,) * (width - len(actual_release)) >= minimum_release + (0,) * (
        width - len(minimum_release)
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_checked(*arguments: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(("git", *arguments), check=True, capture_output=True, text=True)
        return True, result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False, ""


def _file_digest(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _entity_id(entity: Any) -> str:
    return f"{type(entity).__name__}:{getattr(entity, 'name', '<unnamed>')}"


def _criteria_manifest(
    contract: dict[str, Any],
    source: dict[str, Any],
    environment: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, bool]:
    expected_ids = [workload["id"] for workload in contract["workloads"]]
    manifests = [result.get("manifest") for result in results.values()]
    group_manifests = [group for manifest in manifests if manifest for group in manifest["groups"]]
    provenance_complete = bool(group_manifests) and all(
        group["domains"]["provenance_coverage"] == contract["pass_criteria"]["provenance_coverage"]
        for group in group_manifests
    )
    domains_nonempty = bool(group_manifests) and all(
        all(tensor["placement_count"] > 0 for tensor in group["domains"]["tensors"])
        and all(transfer["path_count"] > 0 for transfer in group["domains"]["transfers"])
        for group in group_manifests
    )
    traces_complete = bool(group_manifests) and all(
        tuple(group["group_trace"]) == _GROUP_STAGES and tuple(group["preparation_trace"]) == _PREPARATION_STAGES
        for group in group_manifests
    )
    exclusions_declared = bool(manifests) and all(
        manifest is not None and manifest.get("semantic_exclusions_audited") is True for manifest in manifests
    )
    forbidden_events_absent = bool(manifests) and all(
        manifest is not None
        and set(manifest.get("execution_boundary", {})) == {event.value for event in ExecutionEvent}
        and all(count == 0 for count in manifest["execution_boundary"].values())
        for manifest in manifests
    )
    return {
        "source_identified": bool(source["identified"]),
        "environment_compatible": bool(environment["compatible"]),
        "frozen_denominator_intact": len(expected_ids) == len(set(expected_ids)) and set(results) == set(expected_ids),
        "lifting_success": len(results) == len(expected_ids) and all(result["valid"] for result in results.values()),
        "deterministic_preparation": all(result["deterministic"] for result in results.values()),
        "provenance_coverage": provenance_complete,
        "nonempty_option_domains": domains_nonempty,
        "complete_stage_traces": traces_complete,
        "silent_fallbacks": bool(group_manifests) and all(not group["mapping_fallbacks"] for group in group_manifests),
        "declared_semantic_exclusions": exclusions_declared,
        "forbidden_execution_events_absent": forbidden_events_absent,
    }


def _summary(results: dict[str, Any]) -> dict[str, Any]:
    valid = sum(result["valid"] for result in results.values())
    reasons: dict[str, int] = {}
    for result in results.values():
        for attempt in result["attempts"]:
            reason = attempt["reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
    forbidden_counts = {event.value: 0 for event in ExecutionEvent}
    for result in results.values():
        manifest = result.get("manifest")
        if manifest is None:
            continue
        for event, count in manifest.get("execution_boundary", {}).items():
            forbidden_counts[event] += count
    return {
        "valid_workloads": valid,
        "required_workloads": len(results),
        "lifting_success": valid / len(results),
        "reason_counts": reasons,
        "forbidden_execution_counts": forbidden_counts,
        "tta_constructed": forbidden_counts[ExecutionEvent.TTA_CONSTRUCT.value] > 0,
        "tta_solved": forbidden_counts[ExecutionEvent.TTA_SOLVE.value] > 0,
        "structural_search_run": any(
            forbidden_counts[event.value] > 0
            for event in (ExecutionEvent.STRUCTURAL_EXHAUSTIVE, ExecutionEvent.STRUCTURAL_VARIABLE_ELIMINATION)
        ),
    }


__all__ = ["LiftingError", "LiftingReason", "load_gate2a_contract", "run_real_workload_lifting"]


if __name__ == "__main__":
    if len(sys.argv) != _WORKER_ARGUMENT_COUNT or sys.argv[1] != "--worker-request":
        raise SystemExit("This module is an internal Gate 2A preparation worker.")
    _run_preparation_worker(Path(sys.argv[2]))
