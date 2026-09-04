"""Small, exhaustively enumerable Gate 0 factor graphs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from stream.structural.direct_cost import DirectEvent
from stream.structural.factors import FactorGraph, OwnedEventFactor, PhysicalEvent, Variable
from stream.structural.states import (
    DistributionKind,
    DistributionPlan,
    DistributionTemplate,
    MaterializationMode,
    OperatorState,
    TensorRealization,
)
from stream.workload.tensor_domain import AffineBox, TensorTileDomain

_TENSOR = TensorTileDomain((4, 4), dtype_bytes=2)
_BYTES = 32
_ITERATION = AffineBox((0, 0), (4, 4))
_BLOCK = DistributionTemplate("block", DistributionKind.BLOCK, ("compute",))
_SHARED = DistributionTemplate("shared", DistributionKind.SHARED, ("memory",))


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    producer: str
    consumers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MicroCase:
    graph: FactorGraph
    tensors: tuple[TensorSpec, ...]

    def direct_events(self, assignment):
        events: list[DirectEvent] = []
        for spec in self.tensors:
            producer = assignment[spec.producer]
            realization = assignment[f"q:{spec.name}"]
            plan = assignment[f"d:{spec.name}"]
            if realization.fragments[0].layout != producer.output_layout:
                return None
            expected_plan = "block" if realization.materialization is MaterializationMode.STREAMING else "shared"
            if plan.kind != expected_plan or plan.consumers != spec.consumers:
                return None
            if realization.materialization is MaterializationMode.FULL:
                events.append(DirectEvent(f"{spec.name}:materialize", "materialize", _BYTES))
            for consumer_name in spec.consumers:
                consumer = assignment[consumer_name]
                if realization.materialization is MaterializationMode.FULL:
                    events.append(DirectEvent(f"{spec.name}:read:{consumer_name}", "consume", _BYTES))
                if realization.fragments[0].layout != consumer.output_layout:
                    events.append(DirectEvent(f"{spec.name}:retile:{consumer_name}", "retile", _BYTES))
        return tuple(events)


def _operator_states(name: str, fixed_layout: str | None = None) -> tuple[OperatorState, ...]:
    layouts = {"h": (0, 1), "w": (1, 0)}
    selected = (fixed_layout,) if fixed_layout is not None else ("h", "w")
    return tuple(OperatorState(f"{name}:{key}", _ITERATION, layouts[key], "compute") for key in selected)


def _realizations(name: str) -> tuple[TensorRealization, ...]:
    return (
        TensorRealization(
            f"{name}:stream:h",
            _TENSOR.partition(0, 2, layout=(0, 1)),
            _BLOCK,
            MaterializationMode.STREAMING,
        ),
        TensorRealization(
            f"{name}:stream:w",
            _TENSOR.partition(1, 2, layout=(1, 0)),
            _BLOCK,
            MaterializationMode.STREAMING,
        ),
        TensorRealization(
            f"{name}:full:h",
            _TENSOR.partition(0, 2, layout=(0, 1)),
            _SHARED,
            MaterializationMode.FULL,
        ),
        TensorRealization(
            f"{name}:full:w",
            _TENSOR.partition(1, 2, layout=(1, 0)),
            _SHARED,
            MaterializationMode.FULL,
        ),
    )


def _plans(consumers: tuple[str, ...]) -> tuple[DistributionPlan, ...]:
    return (
        DistributionPlan("block", "block", consumers),
        DistributionPlan("shared", "shared", consumers),
    )


def _event(owner: str, key: str, kind: str, cost: int = _BYTES) -> PhysicalEvent:
    return PhysicalEvent(key, kind, cost, owner)


def _tensor_factors(spec: TensorSpec, domains: dict[str, tuple]) -> tuple[OwnedEventFactor, ...]:
    q_name, d_name = f"q:{spec.name}", f"d:{spec.name}"
    create_name = f"create:{spec.name}"
    create = {}
    for producer, realization in product(domains[spec.producer], domains[q_name]):
        if realization.fragments[0].layout == producer.output_layout:
            create[(producer, realization)] = ()

    distribution_name = f"distribution:{spec.name}"
    distribution = {}
    for realization, plan in product(domains[q_name], domains[d_name]):
        expected_plan = "block" if realization.materialization is MaterializationMode.STREAMING else "shared"
        if plan.kind != expected_plan or plan.consumers != spec.consumers:
            continue
        distribution[(realization, plan)] = (
            (_event(distribution_name, f"{spec.name}:materialize", "materialize"),)
            if realization.materialization is MaterializationMode.FULL
            else ()
        )

    factors = [
        OwnedEventFactor(create_name, (spec.producer, q_name), create),
        OwnedEventFactor(distribution_name, (q_name, d_name), distribution),
    ]
    for consumer_name in spec.consumers:
        factor_name = f"consume:{spec.name}:{consumer_name}"
        table = {}
        for realization, plan, consumer in product(domains[q_name], domains[d_name], domains[consumer_name]):
            expected_plan = "block" if realization.materialization is MaterializationMode.STREAMING else "shared"
            if plan.kind != expected_plan or plan.consumers != spec.consumers:
                continue
            events = []
            if realization.materialization is MaterializationMode.FULL:
                events.append(_event(factor_name, f"{spec.name}:read:{consumer_name}", "consume"))
            if realization.fragments[0].layout != consumer.output_layout:
                events.append(_event(factor_name, f"{spec.name}:retile:{consumer_name}", "retile"))
            table[(realization, plan, consumer)] = tuple(events)
        factors.append(OwnedEventFactor(factor_name, (q_name, d_name, consumer_name), table))
    return tuple(factors)


def build_case(
    edges: tuple[tuple[str, str, tuple[str, ...]], ...],
    *,
    fixed_layouts: dict[str, str] | None = None,
) -> MicroCase:
    fixed_layouts = fixed_layouts or {}
    operator_names = tuple(dict.fromkeys(name for _, producer, consumers in edges for name in (producer, *consumers)))
    tensors = tuple(TensorSpec(*edge) for edge in edges)
    variables = [Variable(name, _operator_states(name, fixed_layouts.get(name))) for name in operator_names]
    for spec in tensors:
        variables.extend(
            (
                Variable(f"q:{spec.name}", _realizations(spec.name)),
                Variable(f"d:{spec.name}", _plans(spec.consumers)),
            )
        )
    domains = {variable.name: variable.domain for variable in variables}
    factors = tuple(factor for spec in tensors for factor in _tensor_factors(spec, domains))
    return MicroCase(FactorGraph(tuple(variables), factors), tensors)


def six_cases() -> dict[str, MicroCase]:
    return {
        "edge": build_case((("t_ab", "A", ("B",)),)),
        "chain": build_case((("t_ab", "A", ("B",)), ("t_bc", "B", ("C",)))),
        "gemm_chain": build_case((("t_ab", "A", ("B",)),), fixed_layouts={"A": "h", "B": "w"}),
        "conv_chain": build_case((("t_ab", "A", ("B",)),), fixed_layouts={"A": "w", "B": "h"}),
        "residual_diamond": build_case((("t_a", "A", ("B", "C")), ("t_b", "B", ("D",)), ("t_c", "C", ("D",)))),
        "fork_join": build_case((("t_a", "A", ("B", "C", "D")), ("t_join", "D", ("E",)))),
    }
