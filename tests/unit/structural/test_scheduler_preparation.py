from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from stream.cost_model.communication_manager import CommunicationManager
from stream.cost_model.steady_state_scheduler import PreparedScheduleProblem, SteadyStateScheduler
from stream.datatypes import LayerDim
from stream.mapping.mapping import Mapping
from stream.opt.allocation.constraint_optimization.tensor_restriction import TensorRestriction


def _scheduler() -> SteadyStateScheduler:
    return SteadyStateScheduler(
        workload=MagicMock(),
        accelerator=MagicMock(),
        mapping=Mapping(),
        fusion_splits={},
        cost_lut=MagicMock(),
    )


def test_prepare_problem_runs_both_transforms_before_ssis_and_timeslots(monkeypatch):
    scheduler = _scheduler()
    initial = scheduler.mapping
    after_pre = Mapping(runtime_args={"stage": "pre"})
    after_update = Mapping(runtime_args={"stage": "updated"})
    after_post = Mapping(runtime_args={"stage": "post"})
    steady_state = MagicMock()
    timeslots = {MagicMock(): 0}
    events: list[str] = []

    def pre(mapping):
        assert mapping is initial
        events.append("pre")
        return after_pre

    def build_transfer_graph():
        assert scheduler.mapping is after_pre
        events.append("transfer")
        return steady_state

    def update_mapping():
        events.append("mapping")
        return after_update

    def post(mapping):
        assert mapping is after_update
        events.append("post")
        return after_post

    monkeypatch.setattr(scheduler, "build_transfer_graph", build_transfer_graph)
    monkeypatch.setattr(scheduler, "update_fusion_splits", lambda: {})
    monkeypatch.setattr(scheduler, "update_mapping", update_mapping)
    monkeypatch.setattr(scheduler, "update_cost_lut", lambda: events.append("cost") or scheduler.cost_lut)
    monkeypatch.setattr(scheduler, "generate_ssis", lambda: events.append("ssis") or {})
    monkeypatch.setattr(scheduler, "calculate_iterations", lambda: 1)
    monkeypatch.setattr(scheduler, "calculate_multiplicities", lambda: {})
    steady_state.get_timeslots.side_effect = lambda mapping: events.append("timeslots") or timeslots

    preparation_stages: list[str] = []
    prepared = scheduler.prepare_problem(
        pre_mapping_transform=pre,
        post_mapping_transform=post,
        preparation_observer=preparation_stages.append,
    )

    assert events == ["pre", "transfer", "mapping", "post", "cost", "ssis", "timeslots"]
    assert prepared.mapping is after_post
    assert scheduler.mapping is after_post
    steady_state.get_timeslots.assert_called_once_with(after_post)
    assert preparation_stages == [
        "pre_mapping_transform",
        "transfer_graph",
        "fusion_splits",
        "mapping",
        "post_mapping_transform",
        "cost_lut",
        "ssis",
        "iterations",
        "multiplicities",
        "timeslots",
    ]


def test_build_tta_passes_prepared_objects_without_reconstruction(monkeypatch):
    scheduler = _scheduler()
    scheduler.ssw = MagicMock()
    scheduler.iterations = 2
    scheduler.ssis = {}
    timeslots = {MagicMock(): 0}
    multiplicities = {MagicMock(): 3}
    restrictions = (TensorRestriction("T", frozenset({"cores:0"})),)
    prepared = PreparedScheduleProblem(timeslots, multiplicities, scheduler.mapping, restrictions)
    captured = {}

    def allocator(workload, slot_of, **kwargs):
        captured.update(workload=workload, slot_of=slot_of, **kwargs)
        return SimpleNamespace(mapping=kwargs["mapping"], slot_of=slot_of)

    monkeypatch.setattr("stream.cost_model.steady_state_scheduler.TransferAndTensorAllocator", allocator)
    result = scheduler.build_tta(prepared)

    assert result.mapping is prepared.mapping
    assert captured["slot_of"] is timeslots
    assert captured["multiplicities"] is multiplicities
    assert captured["mapping"] is prepared.mapping
    assert captured["tensor_restrictions"] is restrictions


def test_build_tta_rejects_a_stale_mapping():
    scheduler = _scheduler()
    scheduler.ssw = MagicMock()
    stale = PreparedScheduleProblem({}, {}, Mapping())

    with pytest.raises(ValueError, match="stale"):
        scheduler.build_tta(stale)


def test_scheduler_forwards_configured_transfer_plan_limit(monkeypatch):
    scheduler = _scheduler()
    scheduler.max_transfer_plans_per_endpoint = 4
    source = MagicMock()
    source_allocation = (MagicMock(),)
    destination_allocation = (MagicMock(),)
    plans = (MagicMock(), MagicMock())
    monkeypatch.setattr(scheduler, "_retrieve_core_allocation", lambda _node: (source_allocation,))
    scheduler.accelerator.communication_manager.get_possible_transfer_plan.return_value = plans

    result = scheduler.determine_possible_transfer_plans(source, (destination_allocation,))

    assert result == plans
    scheduler.accelerator.communication_manager.get_possible_transfer_plan.assert_called_once_with(
        src_allocs=source_allocation,
        dst_allocs=destination_allocation,
        max_plans=4,
    )


@pytest.mark.parametrize("invalid", [True, 1.5, "4", 0])
def test_transfer_plan_limits_fail_closed(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        SteadyStateScheduler(
            MagicMock(),
            MagicMock(),
            Mapping(),
            {},
            MagicMock(),
            max_transfer_plans_per_endpoint=invalid,
        )
    manager = object.__new__(CommunicationManager)
    with pytest.raises(ValueError, match="positive integer"):
        manager.get_possible_transfer_plan([], [], max_plans=invalid)


def test_transfer_tensor_names_do_not_alias_source_graph_names():
    scheduler = _scheduler()
    scheduler._tensor_names_in_use = {"output_1_1"}

    assert scheduler._fresh_transfer_tensor_name("output_1_1") == "output_1_1.__stream1"
    assert scheduler._fresh_transfer_tensor_name("output_1_1") == "output_1_1.__stream2"


def test_transfer_reference_prefers_tensor_relevant_split_over_raw_core_count():
    scheduler = _scheduler()
    scheduler.ssw = MagicMock()
    scheduler.mapping = MagicMock()
    transfer = SimpleNamespace(name="Transfer(shared)")
    relevant = SimpleNamespace(name="relevant")
    unrelated = SimpleNamespace(name="unrelated")
    tensor_dim = LayerDim(position=0, prefix="z")
    other_dim = LayerDim(position=1, prefix="z")
    scheduler.ssw.get_dims.return_value = (tensor_dim,)

    def tiling(node, _mapping):
        return ((tensor_dim, 4),) if node is relevant else ((other_dim, 32),)

    scheduler.ssw.get_unique_dims_inter_core_tiling.side_effect = tiling
    scheduler.mapping.get.side_effect = lambda node: SimpleNamespace(
        resource_allocation=(tuple(range(4 if node is relevant else 32)),)
    )

    assert scheduler._get_tensor_relevant_compute_reference(transfer, [unrelated, relevant]) is relevant


@pytest.mark.parametrize("other_factor", [2, 4])
def test_transfer_reference_rejects_incompatible_consumer_partitions(other_factor):
    scheduler = _scheduler()
    scheduler.ssw = MagicMock()
    scheduler.mapping = MagicMock()
    transfer = SimpleNamespace(name="Transfer(shared)")
    by_height = SimpleNamespace(name="by_height")
    by_width = SimpleNamespace(name="by_width")
    height = LayerDim(position=0, prefix="z")
    width = LayerDim(position=1, prefix="z")
    scheduler.ssw.get_dims.return_value = (height, width)

    def tiling(node, _mapping):
        return ((height, 4),) if node is by_height else ((width, other_factor),)

    scheduler.ssw.get_unique_dims_inter_core_tiling.side_effect = tiling
    scheduler.mapping.get.return_value = SimpleNamespace(resource_allocation=(tuple(range(4)),))

    with pytest.raises(ValueError, match="incompatible tensor-relevant tilings"):
        scheduler._get_tensor_relevant_compute_reference(transfer, [by_height, by_width])
