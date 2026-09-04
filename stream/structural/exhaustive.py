"""Ground-truth enumeration for finite Gate 0 problems."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from stream.structural.factors import FactorGraph, StateValue

Objective = Callable[[Mapping[str, StateValue]], int | None]


@dataclass(frozen=True, slots=True)
class ExhaustiveResult:
    optimum: int
    assignments: tuple[dict[str, StateValue], ...]
    visited: int
    feasible: int


def exhaustive_minimize(graph: FactorGraph, objective: Objective) -> ExhaustiveResult:
    """Enumerate every full assignment; ``None`` marks an illegal assignment."""
    optimum: int | None = None
    best: list[dict[str, StateValue]] = []
    visited = 0
    feasible = 0
    for assignment in graph.assignments():
        visited += 1
        cost = objective(assignment)
        if cost is None:
            continue
        if not isinstance(cost, int):
            raise TypeError("Gate 0 objectives must return integer costs")
        feasible += 1
        if optimum is None or cost < optimum:
            optimum, best = cost, [assignment]
        elif cost == optimum:
            best.append(assignment)
    if optimum is None:
        raise ValueError("finite structural problem has no feasible assignment")
    return ExhaustiveResult(optimum, tuple(best), visited, feasible)
