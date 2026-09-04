"""Exact, paired operator-template restrictions applied before tiling generation.

An operator template is one atomic ``(core group, inter-core split)`` choice.
Keeping the pair explicit is essential: :class:`~stream.mapping.mapping.NodeMapping`
stores resource and tiling domains separately, whose Cartesian product would admit
combinations that were never present in the structural template library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from math import prod
from typing import Any

from stream.datatypes import LayerDim
from stream.mapping.mapping import Mapping, NodeMapping
from stream.workload.affine_access import map_dim_positions
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import ComputationNode
from stream.workload.workload import Workload


class OperatorTemplateCompileError(ValueError):
    """The requested template is outside the declared finite library or is illegal."""


@dataclass(frozen=True, slots=True, order=True)
class OperatorTemplate:
    """One paired computation placement and output-parallel split template."""

    target: str
    core_ids: tuple[int, ...]
    splits: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("operator template target must be non-empty")
        invalid_core_ids = (
            not self.core_ids
            or len(set(self.core_ids)) != len(self.core_ids)
            or any(core < 0 for core in self.core_ids)
        )
        if invalid_core_ids:
            raise ValueError("operator template core IDs must be non-negative, non-empty, and unique")
        positions = [position for position, _ in self.splits]
        if len(set(positions)) != len(positions):
            raise ValueError("operator template split positions must be unique")
        if any(position < 0 or factor <= 1 for position, factor in self.splits):
            raise ValueError("operator template splits require a non-negative position and factor greater than one")
        if prod(factor for _, factor in self.splits) != len(self.core_ids):
            raise ValueError("the split-factor product must equal the selected core-group size")

    @property
    def key(self) -> str:
        split_key = ",".join(f"D{position}={factor}" for position, factor in self.splits) or "none"
        return f"{self.target}|cores:{','.join(map(str, self.core_ids))}|splits:{split_key}"


@dataclass(frozen=True, slots=True)
class OperatorTemplateLibrary:
    """Finite preregistered set from which an assignment may select."""

    templates: tuple[OperatorTemplate, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported operator-template library version: {self.version}")
        keys = [template.key for template in self.templates]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("operator-template library must be non-empty and duplicate-free")

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(template.key for template in self.templates)


@dataclass(frozen=True, slots=True)
class OperatorTemplateAssignment:
    """At most one selected paired template per computation operator."""

    assignment_id: str
    selections: tuple[OperatorTemplate, ...]

    def __post_init__(self) -> None:
        if not self.assignment_id:
            raise ValueError("operator-template assignment ID must be non-empty")
        targets = [template.target for template in self.selections]
        if len(targets) != len(set(targets)):
            raise ValueError("an operator-template assignment may select each target only once")


@dataclass(frozen=True, slots=True)
class CompiledOperatorTemplates:
    """Deterministic result at the pre-tiling compiler seam."""

    mapping: Mapping = field(compare=False, repr=False)
    assignment: OperatorTemplateAssignment

    @property
    def manifest(self) -> dict[str, Any]:
        from stream.structural.stream_contract import canonical_mapping_manifest  # noqa: PLC0415

        return {
            "assignment_id": self.assignment.assignment_id,
            "selections": [template.key for template in sorted(self.assignment.selections)],
            "mapping": canonical_mapping_manifest(self.mapping),
        }

    @property
    def semantic_hash(self) -> str:
        encoded = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode()).hexdigest()


def compile_operator_templates(
    workload: Workload,
    mapping: Mapping,
    assignment: OperatorTemplateAssignment,
    library: OperatorTemplateLibrary,
) -> CompiledOperatorTemplates:
    """Replace selected node mappings with exact singleton paired templates.

    The compiler does not search, infer, or repair a template. Every selection must
    already occur in ``library`` and must be legal for the current workload and the
    baseline core pool. Unselected nodes retain their original mapping unchanged.
    """

    unknown = sorted(template.key for template in assignment.selections if template.key not in library.keys)
    if unknown:
        raise OperatorTemplateCompileError(f"assignment selects templates outside the declared library: {unknown}")

    compiled = _copy_mapping(mapping)
    for template in assignment.selections:
        matches = [node for node in workload.get_computation_nodes() if node.name == template.target]
        if len(matches) != 1:
            raise OperatorTemplateCompileError(f"template target {template.target!r} is not a unique computation node")
        node = matches[0]
        source = mapping.get(node)
        if len(source.resource_allocation) != 1:
            raise OperatorTemplateCompileError(f"{node.name}: baseline mapping must provide exactly one core pool")
        core_by_id = {core.id: core for core in source.resource_allocation[0]}
        if any(core_id not in core_by_id for core_id in template.core_ids):
            raise OperatorTemplateCompileError(f"{node.name}: selected core group is outside the baseline core pool")
        _validate_splits(workload, node, template)
        selected_tiling = tuple(
            (LayerDim(position=position, prefix="d"), factor) for position, factor in template.splits
        )
        compiled.set(
            node,
            NodeMapping(
                resource_allocation=(tuple(core_by_id[core_id] for core_id in template.core_ids),),
                inter_core_tiling=(selected_tiling,) if selected_tiling else (),
                memory_allocation=tuple(source.memory_allocation),
                kernel=source.kernel,
            ),
        )
    return CompiledOperatorTemplates(compiled, assignment)


def _validate_splits(workload: Workload, node: ComputationNode, template: OperatorTemplate) -> None:
    output_positions = {position for output in node.outputs for position in map_dim_positions(node.get_mapping(output))}
    iterator_types = derive_iterator_types(node)
    node_dims = workload.get_dims(node)
    for position, factor in template.splits:
        if position >= len(node_dims) or position not in output_positions:
            raise OperatorTemplateCompileError(f"{node.name}: D{position} is not an output-indexed dimension")
        if iterator_types.get(position) is not IteratorType.PARALLEL:
            raise OperatorTemplateCompileError(f"{node.name}: D{position} is not a parallel dimension")
        extent = workload.get_dimension_size(node_dims[position])
        if extent % factor:
            raise OperatorTemplateCompileError(f"{node.name}: extent {extent} is not divisible by split {factor}")


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
