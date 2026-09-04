"""Independent full-assignment cost semantics for Gate 0."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from stream.structural.factors import FactorGraph, StateValue


@dataclass(frozen=True, slots=True)
class DirectEvent:
    key: str
    kind: str
    cost: int

    def __post_init__(self) -> None:
        if not self.key or not self.kind:
            raise ValueError("direct events require key and kind")
        if not isinstance(self.cost, int) or self.cost < 0:
            raise ValueError("direct-event cost must be a non-negative integer")


DirectEventBuilder = Callable[[Mapping[str, StateValue]], Iterable[DirectEvent] | None]


def direct_events(assignment: Mapping[str, StateValue], builder: DirectEventBuilder) -> tuple[DirectEvent, ...] | None:
    built = builder(assignment)
    if built is None:
        return None
    events = tuple(built)
    keys = [event.key for event in events]
    if len(set(keys)) != len(keys):
        raise ValueError("direct semantics emitted the same physical event more than once")
    return events


def direct_cost(assignment: Mapping[str, StateValue], builder: DirectEventBuilder) -> int | None:
    events = direct_events(assignment, builder)
    return None if events is None else sum(event.cost for event in events)


def assert_factor_accounting(
    graph: FactorGraph,
    assignment: Mapping[str, StateValue],
    builder: DirectEventBuilder,
) -> None:
    """Require factorized and independent semantics to own identical events."""
    reference = direct_events(assignment, builder)
    factorized = graph.factorized_events(assignment)
    if reference is None or factorized is None:
        if reference is not None or factorized is not None:
            raise AssertionError("direct and factor semantics disagree on assignment legality")
        return
    reference_by_key = {event.key: (event.kind, event.cost) for event in reference}
    factorized_by_key = {event.key: (event.kind, event.cost) for event in factorized}
    if reference_by_key != factorized_by_key:
        raise AssertionError(
            f"physical-event accounting mismatch: direct={reference_by_key}, factorized={factorized_by_key}"
        )
