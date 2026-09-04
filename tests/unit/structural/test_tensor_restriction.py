from __future__ import annotations

import pytest
from xdsl.dialects.builtin import bf16
from xdsl.ir.affine import AffineMap

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.hardware.architecture.core import Core
from stream.hardware.architecture.noc.communication_link import CommunicationLink
from stream.mapping.mapping import Mapping, NodeMapping
from stream.opt.allocation.constraint_optimization.tensor_restriction import (
    InvalidTensorRestrictionError,
    TensorRestriction,
    TransferPlanRestriction,
    apply_tensor_restrictions_to_mapping,
    restrict_option_domains,
    tensor_placement_key,
    transfer_plan_key,
)
from stream.workload.node import ComputationNode, TransferNode, TransferType
from stream.workload.tensor import Tensor


def _core(core_id: int) -> Core:
    return Core(core_id=core_id, name=f"core_{core_id}", core_type="compute")


def _case():
    source, middle, target = (_core(index) for index in range(3))
    tensor_in = Tensor.create("T.in", bf16, (8,))
    tensor_out = Tensor.create("T", bf16, (8,))
    transfer = TransferNode(
        name="Transfer(T)",
        inputs=(tensor_in,),
        outputs=(tensor_out,),
        operand_mapping=(AffineMap.identity(1), AffineMap.identity(1)),
        transfer_type=TransferType.COMPUTE_TO_COMPUTE,
    )
    direct_link = CommunicationLink(source, target, bandwidth=32, unit_energy_cost=1)
    first = CommunicationLink(source, middle, bandwidth=32, unit_energy_cost=1)
    second = CommunicationLink(middle, target, bandwidth=32, unit_energy_cost=1)
    direct = MulticastPathPlan((source,), (target,), 1, (direct_link,))
    indirect = MulticastPathPlan((source,), (target,), 2, (first, second))
    placements = ((source,), (target,))
    paths = (direct, indirect)
    restriction = TensorRestriction(
        tensor_out.name,
        frozenset({tensor_placement_key(placements[1])}),
        (
            TransferPlanRestriction(
                transfer.name,
                frozenset({transfer_plan_key(indirect)}),
            ),
        ),
    )
    return tensor_out, transfer, placements, paths, restriction


def test_restriction_is_exact_order_preserving_domain_intersection():
    tensor, transfer, placements, paths, restriction = _case()

    tensor_domains, transfer_domains = restrict_option_domains(
        {tensor: placements},
        {transfer: paths},
        (restriction,),
    )

    assert tensor_domains[tensor] == (placements[1],)
    assert transfer_domains[transfer] == (paths[1],)


def test_no_restrictions_preserve_original_domains_without_aliasing_the_dicts():
    tensor, transfer, placements, paths, _ = _case()
    tensor_input = {tensor: placements}
    transfer_input = {transfer: paths}

    tensor_domains, transfer_domains = restrict_option_domains(tensor_input, transfer_input, ())

    assert tensor_domains == tensor_input and tensor_domains is not tensor_input
    assert transfer_domains == transfer_input and transfer_domains is not transfer_input


def test_path_restriction_filters_mapping_before_timeslots():
    _, transfer, placements, paths, restriction = _case()
    mapping = Mapping({transfer: NodeMapping(resource_allocation=paths, memory_allocation=placements)})

    assert apply_tensor_restrictions_to_mapping(mapping, (restriction,)) is mapping
    assert mapping.get(transfer).resource_allocation == (paths[1],)
    assert mapping.get(transfer).memory_allocation == (placements[1],)


def test_unknown_choice_is_rejected_instead_of_silently_relaxed():
    tensor, transfer, placements, paths, restriction = _case()
    invalid = TensorRestriction(tensor.name, frozenset({"cores:99"}), restriction.allowed_transfer_plans)

    with pytest.raises(InvalidTensorRestrictionError, match="unknown choices"):
        restrict_option_domains({tensor: placements}, {transfer: paths}, (invalid,))


def test_empty_and_duplicate_restriction_domains_fail_closed():
    _, _, placements, _, _ = _case()

    with pytest.raises(ValueError, match="non-empty"):
        TensorRestriction("T", frozenset())
    with pytest.raises(ValueError, match="constrain"):
        TensorRestriction("T")
    duplicate = TensorRestriction("T", frozenset({tensor_placement_key(placements[0])}))
    with pytest.raises(InvalidTensorRestrictionError, match="targets must be unique"):
        restrict_option_domains({}, {}, (duplicate, duplicate))


def test_transfer_must_belong_to_the_restricted_tensor():
    tensor, transfer, placements, paths, restriction = _case()
    unrelated = Tensor.create("unrelated", bf16, (8,))
    invalid = TensorRestriction(
        unrelated.name,
        frozenset({tensor_placement_key(placements[0])}),
        restriction.allowed_transfer_plans,
    )

    with pytest.raises(InvalidTensorRestrictionError, match="not associated"):
        restrict_option_domains(
            {unrelated: placements},
            {transfer: paths},
            (invalid,),
        )


def test_compute_output_placement_is_rejected_from_the_v2_exact_subset():
    tensor, transfer, placements, paths, restriction = _case()
    compute_input = Tensor.create("compute.in", bf16, (8,))
    compute = ComputationNode(
        name="A",
        inputs=(compute_input,),
        outputs=(tensor,),
        operand_mapping=(AffineMap.identity(1), AffineMap.identity(1)),
        type="Elementwise",
    )
    mapping = Mapping(
        {
            compute: NodeMapping(resource_allocation=placements),
            transfer: NodeMapping(resource_allocation=paths),
        }
    )

    with pytest.raises(InvalidTensorRestrictionError, match="requires one transfer producer"):
        apply_tensor_restrictions_to_mapping(mapping, (restriction,))


def test_staging_placement_requires_every_adjacent_transfer_path():
    tensor, transfer, placements, paths, _ = _case()
    second_output = Tensor.create("T.next", bf16, (8,))
    second = TransferNode(
        name="Transfer(T.next)",
        inputs=(tensor,),
        outputs=(second_output,),
        operand_mapping=(AffineMap.identity(1), AffineMap.identity(1)),
        transfer_type=TransferType.COMPUTE_TO_COMPUTE,
    )
    incomplete = TensorRestriction(
        tensor.name,
        frozenset({tensor_placement_key(placements[1])}),
        (
            TransferPlanRestriction(
                transfer.name,
                frozenset({transfer_plan_key(paths[1])}),
            ),
        ),
    )
    mapping = Mapping(
        {
            transfer: NodeMapping(resource_allocation=paths, memory_allocation=placements),
            second: NodeMapping(resource_allocation=paths),
        }
    )

    with pytest.raises(InvalidTensorRestrictionError, match="must cover all adjacent transfers"):
        apply_tensor_restrictions_to_mapping(mapping, (incomplete,))


def test_path_key_distinguishes_link_cost_fields():
    tensor, transfer, placements, paths, restriction = _case()
    path = paths[0]
    changed_link = CommunicationLink(
        path.links_used[0].sender,
        path.links_used[0].receiver,
        bandwidth=path.links_used[0].bandwidth * 2,
        unit_energy_cost=path.links_used[0].unit_energy_cost,
    )
    changed = MulticastPathPlan(path.sources, path.targets, path.total_hops_objective, (changed_link,))

    assert transfer_plan_key(path) != transfer_plan_key(changed)
