"""First-class restrictions over existing TTA tensor and path option domains."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.mapping.mapping import Mapping
from stream.workload.node import HasOutputs
from stream.workload.tensor import Tensor
from stream.workload.workload import TransferNode

if TYPE_CHECKING:
    from collections.abc import Mapping as MappingView

TensorPlacementChoice: TypeAlias = tuple[Core, ...]


class InvalidTensorRestrictionError(ValueError):
    """A restriction does not identify a non-empty subset of an existing domain."""


@dataclass(frozen=True, slots=True)
class TransferPlanRestriction:
    """Allowed path-plan keys for one transfer associated with a tensor."""

    transfer_id: str
    allowed_plans: frozenset[str]

    def __post_init__(self) -> None:
        if not self.transfer_id:
            raise ValueError("transfer_id must be non-empty")
        if not self.allowed_plans:
            raise ValueError("allowed_plans must be non-empty")


@dataclass(frozen=True, slots=True)
class TensorRestriction:
    """Physical option-domain restriction produced from one tensor realization.

    Values are canonical keys rather than allocator variables.  The same immutable
    contract can therefore filter the post-transfer mapping before timeslot
    construction and independently filter TTA's normalized option domains before
    variables are created.
    """

    tensor_id: str
    allowed_placements: frozenset[str] | None = None
    allowed_transfer_plans: tuple[TransferPlanRestriction, ...] = ()

    def __post_init__(self) -> None:
        if not self.tensor_id:
            raise ValueError("tensor_id must be non-empty")
        if self.allowed_placements is None and not self.allowed_transfer_plans:
            raise ValueError("a tensor restriction must constrain placements or transfer plans")
        if self.allowed_placements is not None and not self.allowed_placements:
            raise ValueError("allowed_placements must be non-empty when provided")
        transfer_ids = tuple(item.transfer_id for item in self.allowed_transfer_plans)
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError(f"duplicate transfer restriction for tensor {self.tensor_id!r}")


def tensor_placement_key(choice: TensorPlacementChoice) -> str:
    """Canonical identity of one ordered tensor-placement choice."""

    return "cores:" + ",".join(str(core.id) for core in choice)


def transfer_plan_key(path: MulticastPathPlan) -> str:
    """Canonical identity of a complete multicast path plan."""

    return json.dumps(
        {
            "sources": [_resource_id(core) for core in path.sources],
            "targets": [_resource_id(core) for core in path.targets],
            "total_hops_objective": path.total_hops_objective,
            "links": [
                {
                    "sender": _resource_id(link.sender),
                    "receiver": _resource_id(link.receiver),
                    "bandwidth": link.bandwidth,
                    "unit_energy_cost": link.unit_energy_cost,
                    "bidirectional": link.bidirectional,
                }
                for link in path.links_used
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def restrict_option_domains(
    tensor_domains: MappingView[Tensor, tuple[TensorPlacementChoice, ...]],
    transfer_domains: MappingView[TransferNode, tuple[MulticastPathPlan, ...]],
    restrictions: tuple[TensorRestriction, ...],
) -> tuple[
    dict[Tensor, tuple[TensorPlacementChoice, ...]],
    dict[TransferNode, tuple[MulticastPathPlan, ...]],
]:
    """Return exact intersections with existing normalized TTA option domains."""

    _validate_unique_restriction_targets(restrictions)
    tensors_by_name = _unique_by_name(tuple(tensor_domains), "tensor")
    transfers_by_name = _unique_by_name(tuple(transfer_domains), "transfer")
    restricted_tensors = dict(tensor_domains)
    restricted_transfers = dict(transfer_domains)

    for restriction in restrictions:
        tensor = _resolve(tensors_by_name, restriction.tensor_id, "tensor")
        if restriction.allowed_placements is not None:
            restricted_tensors[tensor] = _intersect_exact(
                tensor_domains[tensor],
                restriction.allowed_placements,
                tensor_placement_key,
                f"tensor {restriction.tensor_id!r} placements",
            )
        for transfer_restriction in restriction.allowed_transfer_plans:
            transfer = _resolve(transfers_by_name, transfer_restriction.transfer_id, "transfer")
            _validate_tensor_transfer_association(tensor, transfer)
            restricted_transfers[transfer] = _intersect_exact(
                transfer_domains[transfer],
                transfer_restriction.allowed_plans,
                transfer_plan_key,
                f"transfer {transfer_restriction.transfer_id!r} plans",
            )
    return restricted_tensors, restricted_transfers


def apply_tensor_restrictions_to_mapping(
    mapping: Mapping,
    restrictions: tuple[TensorRestriction, ...],
) -> Mapping:
    """Filter supported placement/path domains before resource-aware timeslots.

    Gate 1A-v2 deliberately supports nontrivial placement restrictions only
    for staging tensors produced by a ``TransferNode``. Their physical
    placement is the producer transfer's ``memory_allocation``. Every adjacent
    transfer must also be restricted so timeslot construction cannot rely on a
    path that the restricted placement makes impossible.
    """

    _validate_unique_restriction_targets(restrictions)
    transfers = tuple(node for node in mapping.nodes() if isinstance(node, TransferNode))
    transfers_by_name = _unique_by_name(transfers, "transfer")
    for restriction in restrictions:
        adjacent = tuple(
            transfer
            for transfer in transfers
            if any(tensor.name == restriction.tensor_id for tensor in transfer.tensors)
        )
        if not adjacent:
            raise InvalidTensorRestrictionError(
                f"tensor {restriction.tensor_id!r} has no transfer in the post-transfer mapping"
            )
        placement_options: tuple[TensorPlacementChoice, ...] | None = None
        if restriction.allowed_placements is not None:
            producers = tuple(
                node
                for node in mapping.nodes()
                if isinstance(node, HasOutputs) and any(tensor.name == restriction.tensor_id for tensor in node.outputs)
            )
            if len(producers) != 1 or not isinstance(producers[0], TransferNode):
                raise InvalidTensorRestrictionError(
                    f"placement restriction for {restriction.tensor_id!r} requires one transfer producer"
                )
            producer_mapping = mapping.get(producers[0])
            placement_options = _intersect_exact(
                producer_mapping.memory_allocation,
                restriction.allowed_placements,
                tensor_placement_key,
                f"tensor {restriction.tensor_id!r} mapping placements",
            )
            producer_mapping.memory_allocation = placement_options
            restricted_transfer_ids = {item.transfer_id for item in restriction.allowed_transfer_plans}
            adjacent_transfer_ids = {transfer.name for transfer in adjacent}
            if restricted_transfer_ids != adjacent_transfer_ids:
                raise InvalidTensorRestrictionError(
                    f"placement restriction for {restriction.tensor_id!r} must cover all adjacent transfers: "
                    f"expected {sorted(adjacent_transfer_ids)}, got {sorted(restricted_transfer_ids)}"
                )
        for transfer_restriction in restriction.allowed_transfer_plans:
            transfer = _resolve(transfers_by_name, transfer_restriction.transfer_id, "transfer")
            if not any(tensor.name == restriction.tensor_id for tensor in transfer.tensors):
                raise InvalidTensorRestrictionError(
                    f"transfer {transfer.name!r} is not associated with tensor {restriction.tensor_id!r}"
                )
            node_mapping = mapping.get(transfer)
            path_options = _intersect_exact(
                node_mapping.resource_allocation,
                transfer_restriction.allowed_plans,
                transfer_plan_key,
                f"transfer {transfer.name!r} mapping plans",
            )
            if placement_options is not None:
                allowed_placements = {tensor_placement_key(option) for option in placement_options}
                for path in path_options:
                    endpoint_groups = []
                    if any(tensor.name == restriction.tensor_id for tensor in transfer.inputs):
                        endpoint_groups.append(tuple(path.sources))
                    if any(tensor.name == restriction.tensor_id for tensor in transfer.outputs):
                        endpoint_groups.append(tuple(path.targets))
                    if not endpoint_groups or any(
                        tensor_placement_key(group) not in allowed_placements for group in endpoint_groups
                    ):
                        raise InvalidTensorRestrictionError(
                            f"transfer {transfer.name!r} plan is incompatible with tensor "
                            f"{restriction.tensor_id!r} placements"
                        )
            node_mapping.resource_allocation = path_options
    return mapping


def restriction_manifest(restrictions: tuple[TensorRestriction, ...]) -> tuple[dict[str, Any], ...]:
    """Canonical serializable form used by Gate 1A-v2 evidence."""

    return tuple(
        {
            "tensor_id": restriction.tensor_id,
            "allowed_placements": (
                sorted(restriction.allowed_placements) if restriction.allowed_placements is not None else None
            ),
            "allowed_transfer_plans": [
                {
                    "transfer_id": item.transfer_id,
                    "allowed_plans": sorted(item.allowed_plans),
                }
                for item in sorted(restriction.allowed_transfer_plans, key=lambda item: item.transfer_id)
            ],
        }
        for restriction in sorted(restrictions, key=lambda item: item.tensor_id)
    )


def _intersect_exact(options: tuple, allowed: frozenset[str], key, label: str) -> tuple:
    option_by_key = {key(option): option for option in options}
    if len(option_by_key) != len(options):
        raise InvalidTensorRestrictionError(f"{label} contain duplicate canonical option keys")
    unknown = allowed - option_by_key.keys()
    if unknown:
        raise InvalidTensorRestrictionError(f"{label} reference unknown choices: {sorted(unknown)}")
    intersection = tuple(option for option in options if key(option) in allowed)
    if not intersection:
        raise InvalidTensorRestrictionError(f"{label} have an empty intersection")
    return intersection


def _validate_unique_restriction_targets(restrictions: tuple[TensorRestriction, ...]) -> None:
    tensor_ids = tuple(item.tensor_id for item in restrictions)
    if len(set(tensor_ids)) != len(tensor_ids):
        raise InvalidTensorRestrictionError("tensor restriction targets must be unique")
    transfer_ids = []
    for restriction in restrictions:
        transfer_ids.extend(item.transfer_id for item in restriction.allowed_transfer_plans)
    if len(set(transfer_ids)) != len(transfer_ids):
        raise InvalidTensorRestrictionError("transfer restriction targets must be unique")


def _unique_by_name(entities: tuple, label: str) -> dict[str, object]:
    by_name: dict[str, object] = {}
    for entity in entities:
        if entity.name in by_name:
            raise InvalidTensorRestrictionError(f"{label} name {entity.name!r} is not unique")
        by_name[entity.name] = entity
    return by_name


def _resolve(entities: dict[str, object], name: str, label: str):
    try:
        return entities[name]
    except KeyError as exc:
        raise InvalidTensorRestrictionError(f"unknown {label} target {name!r}") from exc


def _validate_tensor_transfer_association(tensor: Tensor, transfer: TransferNode) -> None:
    if tensor not in transfer.tensors:
        raise InvalidTensorRestrictionError(f"transfer {transfer.name!r} is not associated with tensor {tensor.name!r}")


def _resource_id(resource) -> str:
    if isinstance(resource, str):
        return f"str:{resource}"
    if hasattr(resource, "id"):
        return f"{type(resource).__name__}:{resource.id}"
    raise TypeError(f"unsupported path endpoint type: {type(resource).__name__}")
