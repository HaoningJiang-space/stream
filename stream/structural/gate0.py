"""Quantitative correctness gate for the finite structural model."""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import prod
from pathlib import Path

from stream.structural.canonicalize import canonical_assignment
from stream.structural.direct_cost import assert_factor_accounting, direct_cost
from stream.structural.elimination import variable_elimination
from stream.structural.exhaustive import exhaustive_minimize
from stream.structural.micro_dags import DAG_CLASSES, MicroDagCase, random_micro_dag

MIN_CONFIGS_PER_CLASS = 20


@dataclass(frozen=True, slots=True)
class Gate0CaseMetrics:
    dag_class: str
    seed: int
    configuration_digest: str
    assignment_count: int
    feasible_assignment_count: int
    direct_factor_checks: int
    direct_factor_mismatches: int
    exhaustive_optimum: int
    ve_optimum: int
    canonical_optimum_hit: bool
    baseline_template_covered: bool
    induced_width: int
    peak_factor_table_entries: int
    state_compression_rate: float
    illegal_states: int
    exact_eliminations: int
    canonical_equivalences: int
    heuristic_prunes: int
    reduction_accounting_valid: bool
    wall_time_ns: int
    peak_memory_bytes: int

    @property
    def passed(self) -> bool:
        return (
            self.direct_factor_mismatches == 0
            and self.exhaustive_optimum == self.ve_optimum
            and self.canonical_optimum_hit
            and self.baseline_template_covered
            and self.heuristic_prunes == 0
            and self.reduction_accounting_valid
        )


@dataclass(frozen=True, slots=True)
class Gate0Report:
    configs_per_class: int
    base_seed: int
    cases: tuple[Gate0CaseMetrics, ...]

    @property
    def configuration_count(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> bool:
        counts = {dag_class: 0 for dag_class in DAG_CLASSES}
        digests = {dag_class: set() for dag_class in DAG_CLASSES}
        for case in self.cases:
            counts[case.dag_class] += 1
            digests[case.dag_class].add(case.configuration_digest)
        return (
            self.configs_per_class >= MIN_CONFIGS_PER_CLASS
            and all(count >= MIN_CONFIGS_PER_CLASS for count in counts.values())
            and all(len(values) >= MIN_CONFIGS_PER_CLASS for values in digests.values())
            and all(case.passed for case in self.cases)
        )

    def summary(self) -> dict[str, int | float | bool]:
        total_checks = sum(case.direct_factor_checks for case in self.cases)
        total_mismatches = sum(case.direct_factor_mismatches for case in self.cases)
        return {
            "passed": self.passed,
            "dag_classes": len({case.dag_class for case in self.cases}),
            "configuration_count": self.configuration_count,
            "unique_configuration_count": len({case.configuration_digest for case in self.cases}),
            "configs_per_class": self.configs_per_class,
            "assignment_count": sum(case.assignment_count for case in self.cases),
            "direct_factor_checks": total_checks,
            "direct_factor_mismatches": total_mismatches,
            "direct_factor_match_rate": (total_checks - total_mismatches) / total_checks,
            "ve_optimum_match_count": sum(case.exhaustive_optimum == case.ve_optimum for case in self.cases),
            "ve_optimum_match_rate": sum(case.exhaustive_optimum == case.ve_optimum for case in self.cases)
            / self.configuration_count,
            "canonical_optimum_hit_count": sum(case.canonical_optimum_hit for case in self.cases),
            "canonical_optimum_hit_rate": sum(case.canonical_optimum_hit for case in self.cases)
            / self.configuration_count,
            "baseline_coverage_count": sum(case.baseline_template_covered for case in self.cases),
            "baseline_coverage_rate": sum(case.baseline_template_covered for case in self.cases)
            / self.configuration_count,
            "heuristic_prunes": sum(case.heuristic_prunes for case in self.cases),
            "reduction_accounting_count": sum(case.reduction_accounting_valid for case in self.cases),
            "maximum_induced_width": max(case.induced_width for case in self.cases),
            "maximum_factor_table_entries": max(case.peak_factor_table_entries for case in self.cases),
            "minimum_state_compression_rate": min(case.state_compression_rate for case in self.cases),
            "wall_time_ns": sum(case.wall_time_ns for case in self.cases),
            "peak_memory_bytes": max(case.peak_memory_bytes for case in self.cases),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": {
                "minimum_dag_classes": 6,
                "minimum_random_configs_per_class": MIN_CONFIGS_PER_CLASS,
                "required_correctness_rate": 1.0,
                "heuristic_pruning_allowed": False,
            },
            "summary": self.summary(),
            "cases": [asdict(case) | {"passed": case.passed} for case in self.cases],
        }

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _baseline_covered(case: MicroDagCase) -> bool:
    domains = case.graph.domains
    baseline = case.baseline_assignment
    if set(baseline) != set(domains):
        return False
    if any(value not in domains[name] for name, value in baseline.items()):
        return False
    direct = direct_cost(baseline, case.direct_events)
    factorized = case.graph.factorized_cost(baseline)
    return direct is not None and direct == factorized


def _configuration_digest(case: MicroDagCase) -> str:
    return sha256(repr(case.configuration_key()).encode()).hexdigest()


def evaluate_gate0_case(dag_class: str, seed: int) -> Gate0CaseMetrics:
    tracemalloc.start()
    started = time.perf_counter_ns()
    case = random_micro_dag(dag_class, seed)
    assignment_count = prod(len(variable.domain) for variable in case.graph.variables)
    mismatches = 0
    for assignment in case.graph.assignments():
        try:
            assert_factor_accounting(case.graph, assignment, case.direct_events)
        except (AssertionError, ValueError):
            mismatches += 1
    exhaustive = exhaustive_minimize(case.graph, lambda assignment: direct_cost(assignment, case.direct_events))
    eliminated = variable_elimination(case.graph)
    optimum_classes = {canonical_assignment(assignment) for assignment in exhaustive.assignments}
    canonical_hit = canonical_assignment(eliminated.assignment) in optimum_classes
    baseline_covered = _baseline_covered(case)
    wall_time_ns = time.perf_counter_ns() - started
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_table = eliminated.peak_factor_table_entries
    return Gate0CaseMetrics(
        dag_class=dag_class,
        seed=seed,
        configuration_digest=_configuration_digest(case),
        assignment_count=assignment_count,
        feasible_assignment_count=exhaustive.feasible,
        direct_factor_checks=assignment_count,
        direct_factor_mismatches=mismatches,
        exhaustive_optimum=exhaustive.optimum,
        ve_optimum=eliminated.optimum,
        canonical_optimum_hit=canonical_hit,
        baseline_template_covered=baseline_covered,
        induced_width=eliminated.induced_width,
        peak_factor_table_entries=peak_table,
        state_compression_rate=1.0 - peak_table / assignment_count,
        illegal_states=sum(step.illegal_assignments for step in eliminated.steps),
        exact_eliminations=sum(step.exact_eliminations for step in eliminated.steps),
        canonical_equivalences=sum(step.canonical_equivalences for step in eliminated.steps),
        heuristic_prunes=sum(step.heuristic_prunes for step in eliminated.steps),
        reduction_accounting_valid=all(
            step.candidate_assignments
            == step.output_entries
            + step.illegal_assignments
            + step.exact_eliminations
            + step.canonical_equivalences
            + step.heuristic_prunes
            for step in eliminated.steps
        ),
        wall_time_ns=wall_time_ns,
        peak_memory_bytes=peak_memory_bytes,
    )


def run_gate0(configs_per_class: int = 20, base_seed: int = 20260904) -> Gate0Report:
    if configs_per_class <= 0:
        raise ValueError("configs_per_class must be positive")
    cases = tuple(
        evaluate_gate0_case(dag_class, base_seed + class_index * 100_000 + config_index)
        for class_index, dag_class in enumerate(DAG_CLASSES)
        for config_index in range(configs_per_class)
    )
    return Gate0Report(configs_per_class, base_seed, cases)
