"""Deterministic randomized micro-DAGs for the structural Gate 0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from random import Random

from stream.structural.direct_cost import DirectEvent
from stream.structural.factors import FactorGraph, OwnedEventFactor, PhysicalEvent, StateValue, Variable
from stream.structural.states import (
    DistributionKind,
    DistributionPlan,
    DistributionPlanKind,
    DistributionTemplate,
    MaterializationMode,
    OperatorState,
    TensorRealization,
)
from stream.workload.tensor_domain import AffineBox, TensorTileDomain, fragment_overlap_bytes

DAG_CLASSES = ("edge", "chain", "gemm_chain", "conv_chain", "residual_diamond", "fork_join")


@dataclass(frozen=True, slots=True)
class MovementRates:
    materialize: int
    consume: int
    retile: int


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    producer: str
    consumers: tuple[str, ...]
    domain: TensorTileDomain
    rates: MovementRates
    h_parts: int
    w_parts: int


@dataclass(frozen=True, slots=True)
class MicroDagCase:
    dag_class: str
    seed: int
    graph: FactorGraph
    tensors: tuple[TensorSpec, ...]
    operator_names: tuple[str, ...]
    baseline_assignment: Mapping[str, StateValue]

    def configuration_key(self) -> tuple[object, ...]:
        """Return seed-independent semantics used to prove configuration diversity."""
        domains = self.graph.domains
        operator_templates = tuple((name, domains[name]) for name in self.operator_names)
        return operator_templates, self.tensors

    def direct_events(self, assignment: Mapping[str, StateValue]) -> tuple[DirectEvent, ...] | None:
        events = [
            DirectEvent(f"{name}:execute", "compute", assignment[name].integer_cost) for name in self.operator_names
        ]
        for spec in self.tensors:
            producer = assignment[spec.producer]
            realization = assignment[f"q:{spec.name}"]
            plan = assignment[f"d:{spec.name}"]
            if realization.fragments[0].layout != producer.output_layout:
                return None
            expected_plan = (
                DistributionPlanKind.BLOCK
                if realization.materialization is MaterializationMode.STREAMING
                else DistributionPlanKind.SHARED
            )
            if plan.kind is not expected_plan or plan.consumers != spec.consumers:
                return None
            tensor_bytes = spec.domain.full_box.elements * spec.domain.dtype_bytes
            if realization.materialization is MaterializationMode.FULL:
                events.append(
                    DirectEvent(
                        f"{spec.name}:materialize",
                        "materialize",
                        tensor_bytes * spec.rates.materialize,
                    )
                )
            for consumer_name in spec.consumers:
                consumer = assignment[consumer_name]
                if realization.materialization is MaterializationMode.FULL:
                    events.append(
                        DirectEvent(
                            f"{spec.name}:read:{consumer_name}",
                            "consume",
                            tensor_bytes * spec.rates.consume,
                        )
                    )
                if realization.fragments[0].layout != consumer.output_layout:
                    events.append(
                        DirectEvent(
                            f"{spec.name}:retile:{consumer_name}",
                            "retile",
                            tensor_bytes * spec.rates.retile,
                        )
                    )
        return tuple(events)


def _layout_axis(layout: tuple[int, ...]) -> int:
    return 0 if layout == (0, 1) else 1


def _parts(spec: TensorSpec, layout: tuple[int, ...]) -> int:
    return spec.h_parts if _layout_axis(layout) == 0 else spec.w_parts


def _operator_states(
    name: str,
    iteration_tile: AffineBox,
    rng: Random,
    fixed_layout: str | None,
) -> tuple[OperatorState, ...]:
    layouts = {"h": (0, 1), "w": (1, 0)}
    selected = [fixed_layout] if fixed_layout is not None else ["h", "w"]
    if rng.randrange(2):
        selected.reverse()
    return tuple(
        OperatorState(f"{name}:{key}", iteration_tile, layouts[key], "compute", rng.randint(1, 31)) for key in selected
    )


def _realizations(spec: TensorSpec) -> tuple[TensorRealization, ...]:
    block = DistributionTemplate("block", DistributionKind.BLOCK, ("compute",))
    shared = DistributionTemplate("shared", DistributionKind.SHARED, ("memory",))
    layouts = (((0, 1), spec.h_parts), ((1, 0), spec.w_parts))
    return tuple(
        TensorRealization(
            f"{spec.name}:{mode.value}:{'h' if layout == (0, 1) else 'w'}",
            spec.domain.partition(_layout_axis(layout), parts, layout=layout),
            block if mode is MaterializationMode.STREAMING else shared,
            mode,
        )
        for mode, (layout, parts) in product((MaterializationMode.STREAMING, MaterializationMode.FULL), layouts)
    )


def _plans(consumers: tuple[str, ...]) -> tuple[DistributionPlan, ...]:
    return (
        DistributionPlan("block", DistributionPlanKind.BLOCK, consumers),
        DistributionPlan("shared", DistributionPlanKind.SHARED, consumers),
    )


def _owned_event(owner: str, key: str, kind: str, cost: int) -> PhysicalEvent:
    return PhysicalEvent(key, kind, cost, owner)


def _redistribution_bytes(spec: TensorSpec, realization: TensorRealization, layout: tuple[int, ...]) -> int:
    demands = spec.domain.partition(_layout_axis(layout), _parts(spec, layout), layout=layout)
    return sum(
        fragment_overlap_bytes(produced, demanded, spec.domain.dtype_bytes)
        for produced, demanded in product(realization.fragments, demands)
    )


def _operator_factor(name: str, domain: tuple[StateValue, ...]) -> OwnedEventFactor:
    factor_name = f"operator:{name}"
    return OwnedEventFactor(
        factor_name,
        (name,),
        {(state,): (_owned_event(factor_name, f"{name}:execute", "compute", state.integer_cost),) for state in domain},
    )


def _tensor_factors(spec: TensorSpec, domains: Mapping[str, tuple[StateValue, ...]]) -> tuple[OwnedEventFactor, ...]:
    q_name, d_name = f"q:{spec.name}", f"d:{spec.name}"
    create_name = f"create:{spec.name}"
    create = {
        (producer, realization): ()
        for producer, realization in product(domains[spec.producer], domains[q_name])
        if realization.fragments[0].layout == producer.output_layout
    }

    distribution_name = f"distribution:{spec.name}"
    distribution = {}
    tensor_bytes = spec.domain.full_box.elements * spec.domain.dtype_bytes
    for realization, plan in product(domains[q_name], domains[d_name]):
        expected = (
            DistributionPlanKind.BLOCK
            if realization.materialization is MaterializationMode.STREAMING
            else DistributionPlanKind.SHARED
        )
        if plan.kind is not expected or plan.consumers != spec.consumers:
            continue
        distribution[(realization, plan)] = (
            (
                _owned_event(
                    distribution_name,
                    f"{spec.name}:materialize",
                    "materialize",
                    tensor_bytes * spec.rates.materialize,
                ),
            )
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
            expected = (
                DistributionPlanKind.BLOCK
                if realization.materialization is MaterializationMode.STREAMING
                else DistributionPlanKind.SHARED
            )
            if plan.kind is not expected or plan.consumers != spec.consumers:
                continue
            events = []
            if realization.materialization is MaterializationMode.FULL:
                events.append(
                    _owned_event(
                        factor_name,
                        f"{spec.name}:read:{consumer_name}",
                        "consume",
                        tensor_bytes * spec.rates.consume,
                    )
                )
            if realization.fragments[0].layout != consumer.output_layout:
                events.append(
                    _owned_event(
                        factor_name,
                        f"{spec.name}:retile:{consumer_name}",
                        "retile",
                        _redistribution_bytes(spec, realization, consumer.output_layout) * spec.rates.retile,
                    )
                )
            table[(realization, plan, consumer)] = tuple(events)
        factors.append(OwnedEventFactor(factor_name, (q_name, d_name, consumer_name), table))
    return tuple(factors)


def _dag_edges(dag_class: str) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    definitions = {
        "edge": (("t_ab", "A", ("B",)),),
        "chain": (("t_ab", "A", ("B",)), ("t_bc", "B", ("C",))),
        "gemm_chain": (("t_ab", "A", ("B",)),),
        "conv_chain": (("t_ab", "A", ("B",)),),
        "residual_diamond": (("t_a", "A", ("B", "C")), ("t_b", "B", ("D",)), ("t_c", "C", ("D",))),
        "fork_join": (
            ("t_a", "A", ("B", "C", "D")),
            ("t_b", "B", ("E",)),
            ("t_c", "C", ("E",)),
            ("t_d", "D", ("E",)),
        ),
    }
    try:
        return definitions[dag_class]
    except KeyError as error:
        raise ValueError(f"unknown Gate 0 DAG class: {dag_class}") from error


def random_micro_dag(dag_class: str, seed: int) -> MicroDagCase:
    """Build one deterministic randomized finite Gate 0 configuration."""
    rng = Random(seed)
    height, width = rng.randint(2, 5), rng.randint(2, 5)
    dtype_bytes = rng.choice((1, 2, 4))
    if dag_class == "gemm_chain":
        iteration_tile = AffineBox((0, 0, 0), (height, width, rng.randint(2, 5)))
        fixed_layouts = {"A": "h", "B": "w"}
    elif dag_class == "conv_chain":
        iteration_tile = AffineBox((0, 0, 0, 0), (height, width, 3, 3))
        fixed_layouts = {"A": "w", "B": "h"}
    else:
        iteration_tile = AffineBox((0, 0), (height, width))
        fixed_layouts = {"B": "h", "C": "w", "D": "h", "E": "h"} if dag_class == "fork_join" else {}

    edges = _dag_edges(dag_class)
    operator_names = tuple(dict.fromkeys(name for _, producer, consumers in edges for name in (producer, *consumers)))
    tensor_specs = tuple(
        TensorSpec(
            name,
            producer,
            consumers,
            TensorTileDomain((height, width), dtype_bytes),
            MovementRates(rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)),
            rng.randint(1, min(2, height)),
            rng.randint(1, min(2, width)),
        )
        for name, producer, consumers in edges
    )
    variables = [
        Variable(name, _operator_states(name, iteration_tile, rng, fixed_layouts.get(name))) for name in operator_names
    ]
    for spec in tensor_specs:
        variables.extend(
            (Variable(f"q:{spec.name}", _realizations(spec)), Variable(f"d:{spec.name}", _plans(spec.consumers)))
        )
    domains = {variable.name: variable.domain for variable in variables}
    factors = tuple(_operator_factor(name, domains[name]) for name in operator_names) + tuple(
        factor for spec in tensor_specs for factor in _tensor_factors(spec, domains)
    )
    graph = FactorGraph(tuple(variables), factors)

    baseline: dict[str, StateValue] = {
        name: next((state for state in domains[name] if state.output_layout == (0, 1)), domains[name][0])
        for name in operator_names
    }
    for spec in tensor_specs:
        producer_layout = baseline[spec.producer].output_layout
        baseline[f"q:{spec.name}"] = next(
            realization
            for realization in domains[f"q:{spec.name}"]
            if realization.materialization is MaterializationMode.FULL
            and realization.fragments[0].layout == producer_layout
        )
        baseline[f"d:{spec.name}"] = next(
            plan for plan in domains[f"d:{spec.name}"] if plan.kind is DistributionPlanKind.SHARED
        )
    return MicroDagCase(dag_class, seed, graph, tensor_specs, operator_names, baseline)


def six_cases(seed: int = 0) -> dict[str, MicroDagCase]:
    return {dag_class: random_micro_dag(dag_class, seed) for dag_class in DAG_CLASSES}
