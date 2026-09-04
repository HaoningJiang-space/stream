"""Prepare a structurally restricted STREAM problem without invoking TTA."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256
from typing import Any

from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.cost_model.steady_state_scheduler import PreparedScheduleProblem, SteadyStateScheduler
from stream.opt.allocation.constraint_optimization.tensor_restriction import (
    TensorRestriction,
    apply_tensor_restrictions_to_mapping,
    restriction_manifest,
)
from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import TransferAndTensorAllocator
from stream.structural.stream_contract import (
    CompiledStructuralAssignment,
    CompileStage,
    CompileStatus,
    Gate1AEvalConfig,
    LiteralKind,
    PipelineSemanticManifest,
    StructuralMappingContract,
    canonical_mapping_manifest,
    compile_structural_assignment,
    core_group_key,
    tiling_key,
)
from stream.workload.node import ComputationNode
from stream.workload.workload import Node


@dataclass(frozen=True, slots=True)
class PreparedStructuralProblem:
    """The exact input boundary consumed by unchanged TTA."""

    scheduler: SteadyStateScheduler
    compilation: CompiledStructuralAssignment
    schedule_problem: PreparedScheduleProblem

    @property
    def timeslots(self) -> dict[Node, int]:
        return self.schedule_problem.timeslots

    def build_tta(self) -> TransferAndTensorAllocator:
        """Construct TTA through the same handoff used by ``SteadyStateScheduler.run``."""

        tta = self.scheduler.build_tta(self.schedule_problem)
        if tta.mapping is not self.compilation.mapping:
            raise RuntimeError("TTA did not receive the compiled mapping object")
        if tta.slot_of != self.timeslots:
            raise RuntimeError("TTA did not receive the compiled timeslots")
        return tta


@dataclass(frozen=True, slots=True)
class PreparedTensorRestrictedProblem:
    """A production STREAM problem carrying exact tensor/path domain restrictions."""

    scheduler: SteadyStateScheduler
    restrictions: tuple[TensorRestriction, ...]
    schedule_problem: PreparedScheduleProblem
    pipeline_manifest: PipelineSemanticManifest

    @property
    def timeslots(self) -> dict[Node, int]:
        return self.schedule_problem.timeslots

    @property
    def restriction_manifest(self) -> tuple[dict, ...]:
        return restriction_manifest(self.restrictions)

    @property
    def problem_manifest(self) -> dict[str, Any]:
        return {
            "mapping": canonical_mapping_manifest(self.schedule_problem.mapping),
            "pipeline": asdict(self.pipeline_manifest),
            "restrictions": self.restriction_manifest,
        }

    @property
    def problem_hash(self) -> str:
        return _digest(self.problem_manifest)

    def build_tta(self) -> TransferAndTensorAllocator:
        tta = self.scheduler.build_tta(self.schedule_problem)
        if tta.mapping is not self.schedule_problem.mapping:
            raise RuntimeError("TTA did not receive the tensor-restricted mapping object")
        if tta.tensor_restrictions is not self.restrictions:
            raise RuntimeError("TTA did not receive the exact tensor restriction contract")
        return tta


def prepare_structural_problem(
    scheduler: SteadyStateScheduler,
    contract: StructuralMappingContract,
    eval_config: Gate1AEvalConfig,
) -> PreparedStructuralProblem:
    """Run both compiler phases at the production seams, stopping before TTA construction."""

    _validate_eval_config(scheduler, eval_config)
    pre: CompiledStructuralAssignment | None = None
    post: CompiledStructuralAssignment | None = None
    expected_post_update: dict[str, Any] = {}

    def apply_pre(mapping):
        nonlocal pre
        pre = compile_structural_assignment(mapping, contract, eval_config, CompileStage.PRE_TRANSFER)
        expected_post_update.update(_canonical_pre_expectations(pre, contract))
        return pre.mapping

    def apply_post(mapping):
        nonlocal post
        assert pre is not None
        _verify_pre_literals_after_update(scheduler, mapping, pre, contract, expected_post_update)
        post = compile_structural_assignment(
            mapping,
            contract,
            eval_config,
            CompileStage.POST_TRANSFER,
            prior_classifications=pre.classifications,
        )
        return post.mapping

    schedule_problem = scheduler.prepare_problem(
        pre_mapping_transform=apply_pre,
        post_mapping_transform=apply_post,
    )
    assert post is not None
    pipeline_manifest = _pipeline_manifest(scheduler, schedule_problem, eval_config)
    compilation = replace(post, pipeline_manifest=pipeline_manifest)
    scheduler.mapping = compilation.mapping
    schedule_problem = replace(schedule_problem, mapping=compilation.mapping)
    return PreparedStructuralProblem(scheduler, compilation, schedule_problem)


def prepare_tensor_restricted_problem(
    scheduler: SteadyStateScheduler,
    restrictions: tuple[TensorRestriction, ...],
    eval_config: Gate1AEvalConfig,
) -> PreparedTensorRestrictedProblem:
    """Apply path restrictions before timeslots and all restrictions before TTA variables."""

    _validate_eval_config(scheduler, eval_config)

    def apply_paths(mapping):
        return apply_tensor_restrictions_to_mapping(mapping, restrictions)

    schedule_problem = scheduler.prepare_problem(post_mapping_transform=apply_paths)
    schedule_problem = replace(schedule_problem, tensor_restrictions=restrictions)
    return PreparedTensorRestrictedProblem(
        scheduler,
        restrictions,
        schedule_problem,
        _pipeline_manifest(scheduler, schedule_problem, eval_config),
    )


def _canonical_pre_expectations(
    compilation: CompiledStructuralAssignment,
    contract: StructuralMappingContract,
) -> dict[str, Any]:
    """Capture local pre-transfer semantics for translation after graph reconstruction."""

    classifications = {item.literal_id: item for item in compilation.classifications}
    expected: dict[str, Any] = {}
    for literal in contract.pre_transfer_literals:
        if classifications[literal.literal_id].status is not CompileStatus.EXACT:
            continue
        nodes = [node for node in compilation.mapping.nodes() if node.name == literal.target]
        if len(nodes) != 1 or not isinstance(nodes[0], ComputationNode):
            raise RuntimeError(f"EXACT literal {literal.literal_id!r} lost its computation target")
        node = nodes[0]
        node_mapping = compilation.mapping.get(node)
        if literal.kind is LiteralKind.HARDWARE_ZONE:
            expected[literal.literal_id] = frozenset(
                core_group_key(tuple(option)) for option in node_mapping.resource_allocation
            )
        elif literal.kind is LiteralKind.OPERATOR_TILING:
            expected[literal.literal_id] = tuple(node_mapping.inter_core_tiling)
    return expected


def _verify_pre_literals_after_update(
    scheduler: SteadyStateScheduler,
    mapping,
    compilation: CompiledStructuralAssignment,
    contract: StructuralMappingContract,
    expected: dict[str, Any],
) -> None:
    """Reject a false-EXACT if production mapping reconstruction changes a literal."""

    classifications = {item.literal_id: item for item in compilation.classifications}
    for literal in contract.pre_transfer_literals:
        if classifications[literal.literal_id].status is not CompileStatus.EXACT:
            continue
        nodes = [node for node in mapping.nodes() if node.name == literal.target]
        if len(nodes) != 1 or not isinstance(nodes[0], ComputationNode):
            raise RuntimeError(f"EXACT literal {literal.literal_id!r} lost its computation target after update_mapping")
        node_mapping = mapping.get(nodes[0])
        if literal.kind is LiteralKind.HARDWARE_ZONE:
            observed = frozenset(core_group_key(tuple(option)) for option in node_mapping.resource_allocation)
            intended = expected[literal.literal_id]
        else:
            observed = frozenset(tiling_key(option) for option in node_mapping.inter_core_tiling)
            assert scheduler.ssw is not None
            post_dims = scheduler.ssw.get_dims(nodes[0])
            translated = []
            for option in expected[literal.literal_id]:
                translated.append(
                    tuple((dim if "z" in str(dim) else post_dims[dim.position], factor) for dim, factor in option)
                )
            intended = frozenset(tiling_key(option) for option in translated)
        if observed != intended:
            raise RuntimeError(
                f"false EXACT for {literal.literal_id!r}: expected {sorted(intended)}, "
                f"observed {sorted(observed)} after update_mapping"
            )


def prepare_reference_problem(
    scheduler: SteadyStateScheduler,
    eval_config: Gate1AEvalConfig,
) -> PreparedStructuralProblem:
    """Run the unchanged reference pipeline, represented by an empty restriction contract."""

    return prepare_structural_problem(
        scheduler,
        StructuralMappingContract("reference", ()),
        eval_config,
    )


def prepare_uninstrumented_reference_problem(
    scheduler: SteadyStateScheduler,
    eval_config: Gate1AEvalConfig,
) -> PreparedStructuralProblem:
    """Prepare the ordinary production path with no compiler callbacks."""

    _validate_eval_config(scheduler, eval_config)
    schedule_problem = scheduler.prepare_problem()
    compilation = CompiledStructuralAssignment(
        mapping=schedule_problem.mapping,
        classifications=(),
        eval_config=eval_config,
        stage=CompileStage.POST_TRANSFER,
        pipeline_manifest=_pipeline_manifest(scheduler, schedule_problem, eval_config),
    )
    return PreparedStructuralProblem(scheduler, compilation, schedule_problem)


def _pipeline_manifest(
    scheduler: SteadyStateScheduler,
    prepared: PreparedScheduleProblem,
    eval_config: Gate1AEvalConfig,
) -> PipelineSemanticManifest:
    assert scheduler.ssw is not None
    reuse_domains = tuple(
        sorted(
            (_entity_id(tensor), tuple(range(-1, len(scheduler.ssis[tensor].get_applicable_temporal_sizes()))))
            for tensor in scheduler.ssw.tensors
        )
    )
    ssis_domains = tuple(
        sorted(
            (
                _entity_id(entity),
                tuple(
                    (
                        str(variable.dimension),
                        variable.size,
                        variable.effect.name,
                        variable.type.name,
                        variable.reuse.name,
                    )
                    for variable in ssis.variables
                ),
            )
            for entity, ssis in scheduler.ssis.items()
        )
    )
    constraints = scheduler.constraint_selection
    constraint_families = (
        tuple(
            f"{name}={getattr(constraints, name)}"
            for name in ("memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels", "pipelining")
        )
        if constraints is not None
        else ()
    )
    workload_ir = {
        "input": scheduler.workload.get_ir(),
        "steady_state": scheduler.ssw.get_ir(),
    }
    return PipelineSemanticManifest(
        workload_digest=_digest(workload_ir),
        accelerator_digest=_digest(scheduler.accelerator.get_ir()),
        cost_lut_digest=_cost_lut_digest(scheduler.cost_lut),
        timeslots=tuple(sorted((_entity_id(node), slot) for node, slot in prepared.timeslots.items())),
        reuse_option_domains=reuse_domains,
        constraint_families=constraint_families,
        ssis_domains=ssis_domains,
        candidate_correlations=_candidate_correlations(scheduler),
        iterations=scheduler.iterations,
        multiplicities=tuple(
            sorted((_entity_id(node), multiplicity) for node, multiplicity in prepared.multiplicities.items())
        ),
        scheduler_parameters=(("nb_cols_to_use", scheduler.nb_cols_to_use),),
        transfer_context=_transfer_context_manifest(scheduler),
    )


def _cost_lut_digest(cost_lut: CoreCostLUT) -> str:
    entries: list[dict[str, Any]] = []
    for node in sorted(cost_lut.get_nodes(), key=lambda item: item.name):
        for core in sorted(cost_lut.get_cores(node), key=lambda item: item.id):
            cost = cost_lut.get_cost(node, core)
            entries.append(
                {
                    "node": node.name,
                    "core": core.id,
                    "energy_total": cost.energy_total,
                    "latency_total": cost.latency_total,
                    "ideal_cycle": cost.ideal_cycle,
                    "ideal_temporal_cycle": cost.ideal_temporal_cycle,
                    "mem_energy_breakdown": _canonical_json_value(cost.mem_energy_breakdown),
                    "metadata": _canonical_json_value(cost.metadata),
                }
            )
    return _digest(entries)


def _validate_eval_config(scheduler: SteadyStateScheduler, eval_config: Gate1AEvalConfig) -> None:
    if scheduler.backend != eval_config.backend:
        raise ValueError(f"backend mismatch: scheduler={scheduler.backend!r}, EvalConfig={eval_config.backend!r}")
    if eval_config.timeslot_policy != "resource_aware":
        raise ValueError("Gate 1A supports only the production resource_aware timeslot policy")
    if eval_config.time_limit_s is not None or eval_config.solver_params:
        raise ValueError("Gate 1A cannot bind time limits or solver parameters at the current TTA constructor seam")
    if eval_config.cost_lut_digest and eval_config.cost_lut_digest != _cost_lut_digest(scheduler.cost_lut):
        raise ValueError("cost LUT digest does not match EvalConfig")
    constraints = scheduler.constraint_selection
    actual_constraints = (
        tuple(
            name
            for name in ("memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels", "pipelining")
            if constraints is not None and getattr(constraints, name)
        )
        if constraints is not None
        else ()
    )
    if tuple(sorted(eval_config.constraints)) != tuple(sorted(actual_constraints)):
        raise ValueError(
            f"constraint-family mismatch: scheduler={actual_constraints!r}, EvalConfig={eval_config.constraints!r}"
        )


def _digest(value: Any) -> str:
    payload = json.dumps(_canonical_json_value(value), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _entity_id(entity: Any) -> str:
    return f"{type(entity).__name__}:{getattr(entity, 'name', '<unnamed>')}"


def _candidate_correlations(
    scheduler: SteadyStateScheduler,
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...]:
    from stream.structural.stream_contract import path_key  # noqa: PLC0415
    from stream.workload.node import TransferNode  # noqa: PLC0415

    rows = []
    for node in scheduler.mapping.nodes():
        node_mapping = scheduler.mapping.get(node)
        resources = (
            tuple(path_key(option) for option in node_mapping.resource_allocation)
            if isinstance(node, TransferNode)
            else tuple(",".join(str(core.id) for core in option) for option in node_mapping.resource_allocation)
        )
        tilings = tuple(
            ",".join(f"{dim}={factor}" for dim, factor in option) for option in node_mapping.inter_core_tiling
        )
        memories = tuple(",".join(str(core.id) for core in option) for option in node_mapping.memory_allocation)
        rows.append(
            (
                _entity_id(node),
                resources,
                tilings,
                memories,
            )
        )
    return tuple(sorted(rows))


def _transfer_context_manifest(scheduler: SteadyStateScheduler) -> tuple[tuple[str, Any], ...]:
    context = scheduler.transfer_context
    strategies = []
    for strategy in context.namespace_constraints:
        parameters = tuple(
            sorted(
                (name, _canonical_json_value(value))
                for name, value in getattr(strategy, "__dict__", {}).items()
                if not name.startswith("_")
            )
        )
        strategies.append((type(strategy).__name__, parameters))
    return (
        ("offchip_core_id", context.offchip_core_id),
        ("mem_core_ids", tuple(core.id for core in context.mem_cores)),
        ("force_double_buffering", context.force_double_buffering),
        ("force_io_transfers_on_mem_tile", context.force_io_transfers_on_mem_tile),
        ("namespace_constraints", tuple(strategies)),
    )


def _canonical_json_value(value: Any) -> Any:  # noqa: PLR0911
    """Canonicalize manifests without repr/default=str fallbacks."""

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda pair: str(pair[0]))
        return {str(key): _canonical_json_value(item) for key, item in items}
    if isinstance(value, tuple | list):
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [_canonical_json_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _canonical_json_value(getattr(value, item.name)) for item in fields(value)}
    if hasattr(value, "get_ir"):
        return _canonical_json_value(value.get_ir())
    if hasattr(value, "id"):
        return {"type": type(value).__name__, "id": _canonical_json_value(value.id)}
    if hasattr(value, "name"):
        return {"type": type(value).__name__, "name": _canonical_json_value(value.name)}
    raise TypeError(f"cannot canonicalize {type(value).__name__} in a Gate 1A semantic manifest")
