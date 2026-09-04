"""Exact min-sum variable elimination without heuristic pruning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from stream.structural.factors import FactorGraph, OwnedEventFactor, StateValue


@dataclass(frozen=True, slots=True)
class _CostFactor:
    scope: tuple[str, ...]
    table: dict[tuple[StateValue, ...], int]

    def value(self, assignment: dict[str, StateValue]) -> int | None:
        return self.table.get(tuple(assignment[name] for name in self.scope))


@dataclass(frozen=True, slots=True)
class EliminationStep:
    variable: str
    separator: tuple[str, ...]
    input_entries: int
    output_entries: int
    illegal_assignments: int


@dataclass(frozen=True, slots=True)
class EliminationResult:
    optimum: int
    assignment: dict[str, StateValue]
    steps: tuple[EliminationStep, ...]


def _to_cost_factor(factor: OwnedEventFactor) -> _CostFactor:
    return _CostFactor(
        factor.scope, {key: sum(event.cost for event in events) for key, events in factor.events.items()}
    )


def _separator(variable: str, bucket: list[_CostFactor], elimination_order: tuple[str, ...]) -> tuple[str, ...]:
    involved = {name for factor in bucket for name in factor.scope if name != variable}
    return tuple(name for name in elimination_order if name in involved)


def _eliminate_bucket(
    variable: str,
    separator: tuple[str, ...],
    bucket: list[_CostFactor],
    domains: dict[str, tuple[StateValue, ...]],
) -> tuple[dict[tuple[StateValue, ...], int], dict[tuple[StateValue, ...], StateValue], int]:
    output: dict[tuple[StateValue, ...], int] = {}
    choices: dict[tuple[StateValue, ...], StateValue] = {}
    illegal = 0
    separator_values = product(*(domains[name] for name in separator)) if separator else [()]
    for values in separator_values:
        partial = dict(zip(separator, values, strict=True))
        legal_candidates: list[tuple[int, StateValue]] = []
        for candidate in domains[variable]:
            assignment = {**partial, variable: candidate}
            costs = [factor.value(assignment) for factor in bucket]
            if any(cost is None for cost in costs):
                illegal += 1
                continue
            legal_candidates.append((sum(cost for cost in costs if cost is not None), candidate))
        if legal_candidates:
            best_cost, best_value = min(legal_candidates, key=lambda item: item[0])
            output[values] = best_cost
            choices[values] = best_value
    return output, choices, illegal


def variable_elimination(graph: FactorGraph, order: tuple[str, ...] | None = None) -> EliminationResult:
    """Solve the finite factor graph exactly by bucket elimination.

    Missing factor-table entries are illegal assignments. The implementation
    performs no beam search, dominance, approximation, or value truncation.
    """
    domains = graph.domains
    elimination_order = order or tuple(variable.name for variable in graph.variables)
    if len(set(elimination_order)) != len(elimination_order) or set(elimination_order) != set(domains):
        raise ValueError("elimination order must contain every graph variable exactly once")

    factors = [_to_cost_factor(factor) for factor in graph.factors]
    backpointers: list[tuple[str, tuple[str, ...], dict[tuple[StateValue, ...], StateValue]]] = []
    steps: list[EliminationStep] = []

    for variable in elimination_order:
        bucket = [factor for factor in factors if variable in factor.scope]
        factors = [factor for factor in factors if variable not in factor.scope]
        separator = _separator(variable, bucket, elimination_order)
        output, choices, illegal = _eliminate_bucket(variable, separator, bucket, domains)

        if not output:
            raise ValueError(f"eliminating {variable} removed every assignment")
        factors.append(_CostFactor(separator, output))
        backpointers.append((variable, separator, choices))
        steps.append(
            EliminationStep(
                variable=variable,
                separator=separator,
                input_entries=sum(len(factor.table) for factor in bucket),
                output_entries=len(output),
                illegal_assignments=illegal,
            )
        )

    optimum = sum(factor.table[()] for factor in factors)
    assignment: dict[str, StateValue] = {}
    for variable, separator, choices in reversed(backpointers):
        key = tuple(assignment[name] for name in separator)
        assignment[variable] = choices[key]
    return EliminationResult(optimum, assignment, tuple(steps))
