from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from stream.cost_model.steady_state_scheduler import PreparedScheduleProblem, SteadyStateScheduler
from stream.mapping.mapping import Mapping


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

    prepared = scheduler.prepare_problem(pre_mapping_transform=pre, post_mapping_transform=post)

    assert events == ["pre", "transfer", "mapping", "post", "cost", "ssis", "timeslots"]
    assert prepared.mapping is after_post
    assert scheduler.mapping is after_post
    steady_state.get_timeslots.assert_called_once_with(after_post)


def test_build_tta_passes_prepared_objects_without_reconstruction(monkeypatch):
    scheduler = _scheduler()
    scheduler.ssw = MagicMock()
    scheduler.iterations = 2
    scheduler.ssis = {}
    timeslots = {MagicMock(): 0}
    multiplicities = {MagicMock(): 3}
    prepared = PreparedScheduleProblem(timeslots, multiplicities, scheduler.mapping)
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


def test_build_tta_rejects_a_stale_mapping():
    scheduler = _scheduler()
    scheduler.ssw = MagicMock()
    stale = PreparedScheduleProblem({}, {}, Mapping())

    with pytest.raises(ValueError, match="stale"):
        scheduler.build_tta(stale)
