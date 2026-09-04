"""Gate 1A structural-to-STREAM compiler contract.

The compiler only intersects already-generated STREAM candidate sets. It never
manufactures an option or silently approximates a structural literal. This
makes an ``EXACT`` classification mean both no relaxation and no unintended
restriction relative to the reference pipeline candidate set.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from importlib.resources import files
from typing import Any

from xdsl.ir.affine import AffineMap

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping, NodeMapping
from stream.workload.node import ComputationNode, TransferNode


class CompileStatus(StrEnum):
    EXACT = "EXACT"
    UNSUPPORTED = "UNSUPPORTED"


class UnsupportedReason(StrEnum):
    NO_PIPELINE_LITERAL = "UNSUPPORTED_NO_PIPELINE_LITERAL"
    NO_EXACT_REUSE_LITERAL = "UNSUPPORTED_NO_EXACT_REUSE_LITERAL"
    NO_TENSOR_PLACEMENT_FILTER = "UNSUPPORTED_NO_TENSOR_PLACEMENT_FILTER"
    NO_LAYOUT_CONSTRAINT = "UNSUPPORTED_NO_LAYOUT_CONSTRAINT"
    NO_TRANSFER_PATH_STAGE = "UNSUPPORTED_NO_TRANSFER_PATH_STAGE"
    INCOMPATIBLE_TILING_IR = "UNSUPPORTED_INCOMPATIBLE_TILING_IR"


class IllegalStructuralAssignmentError(ValueError):
    """A literal intersection removed every legal option from its target domain."""


class LiteralKind(StrEnum):
    OPERATOR_TILING = "inter_core_tiling"
    HARDWARE_ZONE = "hardware_zone"
    TENSOR_PLACEMENT = "tensor_placement"
    OUTPUT_LAYOUT = "output_layout"
    MATERIALIZATION = "materialization"
    DISTRIBUTION = "distribution"
    DISTRIBUTION_PLAN = "distribution_plan"
    TRANSFER_PATH = "transfer_path"
    EXACT_REUSE = "exact_reuse"


class CompileStage(StrEnum):
    PRE_TRANSFER = "pre_transfer"
    POST_TRANSFER = "post_transfer"


@dataclass(frozen=True, slots=True)
class StructuralLiteral:
    """One finite-state restriction expressed as allowed canonical option keys."""

    literal_id: str
    kind: LiteralKind
    target: str
    allowed: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.literal_id or not self.target:
            raise ValueError("literal_id and target must be non-empty")
        if len(set(self.allowed)) != len(self.allowed):
            raise ValueError(f"literal {self.literal_id!r} contains duplicate allowed values")


@dataclass(frozen=True, slots=True)
class StructuralMappingContract:
    """Versioned literal set compiled at the two production-pipeline seams."""

    assignment_id: str
    literals: tuple[StructuralLiteral, ...]
    intended_space_version: int = 1

    def __post_init__(self) -> None:
        if not self.assignment_id:
            raise ValueError("assignment_id must be non-empty")
        if self.intended_space_version != 1:
            raise ValueError(f"unsupported intended-space version: {self.intended_space_version}")
        literal_ids = [literal.literal_id for literal in self.literals]
        if len(literal_ids) != len(set(literal_ids)):
            raise ValueError("literal IDs must be unique within one structural assignment")

    @property
    def pre_transfer_literals(self) -> tuple[StructuralLiteral, ...]:
        kinds = {LiteralKind.HARDWARE_ZONE, LiteralKind.OPERATOR_TILING}
        return tuple(literal for literal in self.literals if literal.kind in kinds)

    @property
    def post_transfer_literals(self) -> tuple[StructuralLiteral, ...]:
        return tuple(literal for literal in self.literals if literal not in self.pre_transfer_literals)


@dataclass(frozen=True, slots=True)
class LiteralClassification:
    literal_id: str
    status: CompileStatus
    stage: CompileStage | None = None
    reason: UnsupportedReason | None = None

    def __post_init__(self) -> None:
        if (self.status is CompileStatus.EXACT) != (self.reason is None):
            raise ValueError("EXACT has no reason; UNSUPPORTED requires a reason")
        if self.status is CompileStatus.EXACT and self.stage is None:
            raise ValueError("EXACT literals require a compiler stage")


@dataclass(frozen=True, slots=True)
class Gate1AEvalConfig:
    """Configuration fields that must remain equal in baseline/candidate comparisons."""

    backend: str
    constraints: tuple[str, ...] = ()
    cost_lut_digest: str = ""
    timeslot_policy: str = "resource_aware"
    time_limit_s: int | None = None
    solver_params: tuple[tuple[str, str], ...] = ()
    status_policy: tuple[str, ...] = ("OPTIMAL",)


@dataclass(frozen=True, slots=True)
class PipelineSemanticManifest:
    """Canonical downstream context required for a baseline round-trip proof."""

    workload_digest: str
    accelerator_digest: str
    cost_lut_digest: str
    timeslots: tuple[tuple[str, int], ...]
    reuse_option_domains: tuple[tuple[str, tuple[int, ...]], ...]
    constraint_families: tuple[str, ...]
    ssis_domains: tuple[tuple[str, tuple[tuple[str, int, str, str, str], ...]], ...] = ()
    candidate_correlations: tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = ()
    iterations: int = 0
    multiplicities: tuple[tuple[str, int], ...] = ()
    scheduler_parameters: tuple[tuple[str, Any], ...] = ()
    transfer_context: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledStructuralAssignment:
    """A deterministic, inspectable restricted mapping at one compiler stage."""

    mapping: Mapping = field(compare=False, repr=False)
    classifications: tuple[LiteralClassification, ...]
    eval_config: Gate1AEvalConfig
    stage: CompileStage
    pipeline_manifest: PipelineSemanticManifest | None = None

    @property
    def status(self) -> CompileStatus:
        if all(item.status is CompileStatus.EXACT for item in self.classifications):
            return CompileStatus.EXACT
        return CompileStatus.UNSUPPORTED

    def semantic_manifest(self) -> dict[str, Any]:
        """Compiler artifact, including diagnostic classification metadata."""

        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "classifications": [
                {
                    "literal_id": item.literal_id,
                    "status": item.status.value,
                    "stage": item.stage.value if item.stage else None,
                    "reason": item.reason.value if item.reason else None,
                }
                for item in sorted(self.classifications, key=lambda item: item.literal_id)
            ],
            "eval_config": _jsonable(self.eval_config),
            "mapping": canonical_mapping_manifest(self.mapping),
            "pipeline": _jsonable(self.pipeline_manifest),
        }

    @property
    def semantic_hash(self) -> str:
        payload = json.dumps(self.semantic_manifest(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()

    def problem_manifest(self) -> dict[str, Any]:
        """Downstream problem semantics, excluding compiler-only metadata."""

        return {
            "eval_config": _jsonable(self.eval_config),
            "mapping": canonical_mapping_manifest(self.mapping),
            "pipeline": _jsonable(self.pipeline_manifest),
        }

    @property
    def problem_hash(self) -> str:
        payload = json.dumps(self.problem_manifest(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()


def load_intended_space() -> dict[str, Any]:
    """Load the immutable Gate 1A v1 coverage denominator."""

    resource = files("stream.structural.contracts").joinpath("gate1a_intended_space_v1.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def compile_structural_assignment(
    mapping: Mapping,
    contract: StructuralMappingContract,
    eval_config: Gate1AEvalConfig,
    stage: CompileStage,
    *,
    prior_classifications: tuple[LiteralClassification, ...] = (),
    pipeline_manifest: PipelineSemanticManifest | None = None,
) -> CompiledStructuralAssignment:
    """Compile one phase without changing transfer construction, timeslots, or TTA semantics.

    The caller runs the ordinary STREAM transfer-graph and ``update_mapping``
    steps between the two calls. The post-transfer result carries the pre-stage
    evidence forward, so assignment-level status is computed over every literal.
    """

    if stage is CompileStage.PRE_TRANSFER:
        if prior_classifications:
            raise ValueError("pre-transfer compilation cannot have prior classifications")
        return compile_pre_transfer(
            mapping,
            contract.pre_transfer_literals,
            eval_config,
            pipeline_manifest=pipeline_manifest,
        )
    return compile_post_transfer(
        mapping,
        contract.post_transfer_literals,
        eval_config,
        prior_classifications=prior_classifications,
        pipeline_manifest=pipeline_manifest,
    )


def core_group_key(group: tuple[Core, ...]) -> str:
    return "cores:" + ",".join(str(core.id) for core in group)


def tiling_key(tiling: tuple[tuple[Any, int], ...]) -> str:
    return "tiling:" + ",".join(f"{dim}={factor}" for dim, factor in tiling)


def path_key(path: MulticastPathPlan) -> str:
    # Preserve every dataclass field, including link order.  This is a semantic
    # identity key, not merely a route-equivalence approximation.
    return json.dumps(
        {
            "sources": [_resource_id(core) for core in path.sources],
            "targets": [_resource_id(core) for core in path.targets],
            "total_hops_objective": path.total_hops_objective,
            "links": [(_resource_id(link.sender), _resource_id(link.receiver)) for link in path.links_used],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _resource_id(resource: Any) -> str:
    if isinstance(resource, str):
        return f"str:{resource}"
    if hasattr(resource, "id"):
        return f"{type(resource).__name__}:{resource.id}"
    raise TypeError(f"unsupported path endpoint type: {type(resource).__name__}")


def compile_pre_transfer(
    mapping: Mapping,
    literals: tuple[StructuralLiteral, ...],
    eval_config: Gate1AEvalConfig,
    *,
    pipeline_manifest: PipelineSemanticManifest | None = None,
) -> CompiledStructuralAssignment:
    """Apply exact operator filters before ``build_transfer_graph``."""

    compiled = _copy_mapping(mapping)
    classifications: list[LiteralClassification] = []
    for literal in literals:
        if literal.kind is LiteralKind.HARDWARE_ZONE:
            matches = _matching_options(
                compiled, literal, "resource_allocation", core_group_key, ComputationNode
            )
            if len(matches) != 1:
                classifications.append(_unsupported(literal))
                continue
            _set_node_options(compiled, literal, "resource_allocation", matches, ComputationNode)
            classifications.append(_exact(literal, CompileStage.PRE_TRANSFER))
        elif literal.kind is LiteralKind.OPERATOR_TILING:
            matches = _matching_options(compiled, literal, "inter_core_tiling", tiling_key, ComputationNode)
            if not all(option == matches[0] for option in matches):
                classifications.append(_unsupported(literal))
                continue
            _set_node_options(compiled, literal, "inter_core_tiling", matches, ComputationNode)
            classifications.append(_exact(literal, CompileStage.PRE_TRANSFER))
        else:
            classifications.append(_unsupported(literal))
    return CompiledStructuralAssignment(
        mapping=compiled,
        classifications=tuple(classifications),
        eval_config=eval_config,
        stage=CompileStage.PRE_TRANSFER,
        pipeline_manifest=pipeline_manifest,
    )


def compile_post_transfer(
    mapping: Mapping,
    literals: tuple[StructuralLiteral, ...],
    eval_config: Gate1AEvalConfig,
    *,
    prior_classifications: tuple[LiteralClassification, ...] = (),
    pipeline_manifest: PipelineSemanticManifest | None = None,
) -> CompiledStructuralAssignment:
    """Apply exact path filters after ``update_mapping`` and before SSIS/timeslots."""

    compiled = _copy_mapping(mapping)
    classifications = list(prior_classifications)
    for literal in literals:
        if literal.kind is LiteralKind.TRANSFER_PATH:
            _filter_node_options(compiled, literal, "resource_allocation", path_key, TransferNode)
            classifications.append(_exact(literal, CompileStage.POST_TRANSFER))
        else:
            classifications.append(_unsupported(literal))
    return CompiledStructuralAssignment(
        mapping=compiled,
        classifications=tuple(classifications),
        eval_config=eval_config,
        stage=CompileStage.POST_TRANSFER,
        pipeline_manifest=pipeline_manifest,
    )


def canonical_mapping_manifest(mapping: Mapping) -> dict[str, Any]:
    """Canonical candidate-set manifest used for semantic, not backend, hashing."""

    nodes: dict[str, Any] = {}
    for node in sorted(mapping.nodes(), key=lambda item: (item.name, type(item).__name__)):
        node_mapping = mapping.get(node)
        resources: list[str] = []
        for option in node_mapping.resource_allocation:
            if isinstance(node, TransferNode):
                resources.append(path_key(option))
            else:
                resources.append(core_group_key(tuple(option)))
        nodes[node.name] = {
            "node_type": type(node).__name__,
            "resource_options": sorted(resources),
            "tiling_options": sorted(tiling_key(option) for option in node_mapping.inter_core_tiling),
            "memory_options": sorted(core_group_key(tuple(option)) for option in node_mapping.memory_allocation),
            "kernel": _kernel_manifest(node_mapping.kernel),
        }
    return {
        "nodes": nodes,
        "fused_groups": sorted(group.name for group in mapping.fused_groups),
        "runtime_args": _jsonable(dict(sorted(mapping.runtime_args.items()))),
    }


def _copy_mapping(mapping: Mapping) -> Mapping:
    copied = Mapping(fused_groups=mapping.fused_groups, runtime_args=dict(mapping.runtime_args))
    for node, node_mapping in mapping.items():
        copied.set(
            node,
            NodeMapping(
                resource_allocation=tuple(node_mapping.resource_allocation),
                inter_core_tiling=tuple(node_mapping.inter_core_tiling),
                memory_allocation=tuple(node_mapping.memory_allocation),
                kernel=node_mapping.kernel,
            ),
        )
    return copied


def _kernel_manifest(kernel: Any) -> Any:
    if kernel is None:
        return None
    layouts = kernel.operand_layouts() if hasattr(kernel, "operand_layouts") else ()
    return {
        "type": type(kernel).__name__,
        "operand_layouts": [str(layout) for layout in layouts],
    }


def _filter_node_options(
    mapping: Mapping,
    literal: StructuralLiteral,
    attribute: str,
    key: Callable[[Any], str],
    expected_node_type: type,
) -> tuple[str, ...]:
    options = _matching_options(mapping, literal, attribute, key, expected_node_type)
    _set_node_options(mapping, literal, attribute, options, expected_node_type)
    return tuple(key(option) for option in options)


def _matching_options(
    mapping: Mapping,
    literal: StructuralLiteral,
    attribute: str,
    key: Callable[[Any], str],
    expected_node_type: type,
) -> tuple[Any, ...]:
    matches = [node for node in mapping.nodes() if node.name == literal.target]
    if len(matches) != 1 or not isinstance(matches[0], expected_node_type):
        raise ValueError(
            f"literal {literal.literal_id!r} target {literal.target!r} is not a unique {expected_node_type.__name__}"
        )
    node = matches[0]
    node_mapping = mapping.get(node)
    options = tuple(option for option in getattr(node_mapping, attribute) if key(option) in literal.allowed)
    if not options:
        raise IllegalStructuralAssignmentError(
            f"literal {literal.literal_id!r} leaves target {literal.target!r} with an empty {attribute} domain"
        )
    return options


def _set_node_options(
    mapping: Mapping,
    literal: StructuralLiteral,
    attribute: str,
    options: tuple[Any, ...],
    expected_node_type: type,
) -> None:
    matches = [node for node in mapping.nodes() if node.name == literal.target]
    if len(matches) != 1 or not isinstance(matches[0], expected_node_type):
        raise ValueError(
            f"literal {literal.literal_id!r} target {literal.target!r} is not a unique {expected_node_type.__name__}"
        )
    setattr(mapping.get(matches[0]), attribute, options)


def _exact(literal: StructuralLiteral, stage: CompileStage) -> LiteralClassification:
    return LiteralClassification(literal.literal_id, CompileStatus.EXACT, stage=stage)


def _unsupported(literal: StructuralLiteral) -> LiteralClassification:
    reason_by_kind = {
        LiteralKind.TENSOR_PLACEMENT: UnsupportedReason.NO_TENSOR_PLACEMENT_FILTER,
        LiteralKind.OUTPUT_LAYOUT: UnsupportedReason.NO_LAYOUT_CONSTRAINT,
        LiteralKind.MATERIALIZATION: UnsupportedReason.NO_PIPELINE_LITERAL,
        LiteralKind.DISTRIBUTION: UnsupportedReason.NO_TENSOR_PLACEMENT_FILTER,
        LiteralKind.DISTRIBUTION_PLAN: UnsupportedReason.NO_TRANSFER_PATH_STAGE,
        LiteralKind.EXACT_REUSE: UnsupportedReason.NO_EXACT_REUSE_LITERAL,
        LiteralKind.TRANSFER_PATH: UnsupportedReason.NO_TRANSFER_PATH_STAGE,
        LiteralKind.OPERATOR_TILING: UnsupportedReason.INCOMPATIBLE_TILING_IR,
        LiteralKind.HARDWARE_ZONE: UnsupportedReason.INCOMPATIBLE_TILING_IR,
    }
    return LiteralClassification(literal.literal_id, CompileStatus.UNSUPPORTED, reason=reason_by_kind[literal.kind])


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, AffineMap):
        return {"type": "AffineMap", "value": str(value)}
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__} in a Gate 1A semantic manifest")
