"""Tensor-centric integer factors with exact physical-event ownership."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from itertools import product
from typing import TypeAlias

StateValue: TypeAlias = Hashable
Assignment: TypeAlias = Mapping[str, StateValue]


@dataclass(frozen=True, slots=True)
class Variable:
    name: str
    domain: tuple[StateValue, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.domain:
            raise ValueError("variables require a name and non-empty domain")
        if len(set(self.domain)) != len(self.domain):
            raise ValueError(f"variable {self.name} contains duplicate states")


@dataclass(frozen=True, slots=True)
class PhysicalEvent:
    """One integer-cost physical event, owned by exactly one factor."""

    key: str
    kind: str
    cost: int
    owner: str

    def __post_init__(self) -> None:
        if not self.key or not self.kind or not self.owner:
            raise ValueError("physical events require key, kind, and owner")
        if not isinstance(self.cost, int) or self.cost < 0:
            raise ValueError("physical-event cost must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class OwnedEventFactor:
    """A finite factor table whose entries are owned physical events.

    A missing table entry is an illegal local assignment. Event keys must be
    globally unique for a full assignment; :class:`FactorGraph` enforces that
    invariant when evaluating the factorization.
    """

    name: str
    scope: tuple[str, ...]
    events: Mapping[tuple[StateValue, ...], tuple[PhysicalEvent, ...]]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("factors require a name")
        if len(set(self.scope)) != len(self.scope):
            raise ValueError(f"factor {self.name} repeats a variable in its scope")
        for key, events in self.events.items():
            if len(key) != len(self.scope):
                raise ValueError(f"factor {self.name} has a table key with the wrong arity")
            if any(event.owner != self.name for event in events):
                raise ValueError(f"factor {self.name} contains an event owned by another factor")

    def events_for(self, assignment: Assignment) -> tuple[PhysicalEvent, ...] | None:
        key = tuple(assignment[name] for name in self.scope)
        return self.events.get(key)

    def cost_for(self, assignment: Assignment) -> int | None:
        events = self.events_for(assignment)
        return None if events is None else sum(event.cost for event in events)


@dataclass(frozen=True, slots=True)
class FactorGraph:
    variables: tuple[Variable, ...]
    factors: tuple[OwnedEventFactor, ...]

    def __post_init__(self) -> None:
        names = [variable.name for variable in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("factor graph contains duplicate variable names")
        factor_names = [factor.name for factor in self.factors]
        if len(set(factor_names)) != len(factor_names):
            raise ValueError("factor graph contains duplicate factor names")
        unknown = {name for factor in self.factors for name in factor.scope} - set(names)
        if unknown:
            raise ValueError(f"factor scopes contain unknown variables: {sorted(unknown)}")

    @property
    def domains(self) -> dict[str, tuple[StateValue, ...]]:
        return {variable.name: variable.domain for variable in self.variables}

    def assignments(self):
        names = tuple(variable.name for variable in self.variables)
        for values in product(*(variable.domain for variable in self.variables)):
            yield dict(zip(names, values, strict=True))

    def factorized_events(self, assignment: Assignment) -> tuple[PhysicalEvent, ...] | None:
        emitted: list[PhysicalEvent] = []
        seen: set[str] = set()
        for factor in self.factors:
            events = factor.events_for(assignment)
            if events is None:
                return None
            for event in events:
                if event.key in seen:
                    raise ValueError(f"physical event {event.key!r} is charged more than once")
                seen.add(event.key)
                emitted.append(event)
        return tuple(emitted)

    def factorized_cost(self, assignment: Assignment) -> int | None:
        events = self.factorized_events(assignment)
        return None if events is None else sum(event.cost for event in events)
