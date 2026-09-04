"""Evidence records for the three Gate 1A conformance layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from typing import TYPE_CHECKING, TypeAlias

from stream.structural.stream_contract import (
    CompiledStructuralAssignment,
    CompileStatus,
    LiteralClassification,
    StructuralLiteral,
    canonical_mapping_manifest,
    core_group_key,
    load_intended_space,
    path_key,
    tiling_key,
)
from stream.workload.node import ComputationNode, TransferNode

if TYPE_CHECKING:
    from stream.opt.allocation.constraint_optimization.transfer_and_tensor_allocation import TransferAndTensorAllocator

SemanticSolution: TypeAlias = tuple[tuple[str, str], ...]


class Gate1AVerdict(StrEnum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    NARROW = "NARROW"
    FAIL = "FAIL"


@dataclass(frozen=True, order=True, slots=True)
class AuditKey:
    """Unique denominator key; evidence may not be substituted across assignments."""

    dag_class: str
    assignment_id: str
    literal_id: str


@dataclass(frozen=True, slots=True)
class AssignmentAudit:
    """All mandatory evidence for one deterministic structural assignment."""

    dag_class: str
    assignment_id: str
    classifications: tuple[LiteralClassification, ...]
    candidate_audits: tuple[CandidateSetAudit, ...]
    witness_audits: tuple[ViolationWitnessAudit, ...]
    solution_set_audit: SolutionSetAudit
    problem_hashes: tuple[str, ...]
    baseline_round_trip_ok: bool
    silent_relaxations: int = 0
    coverage_cell: tuple[tuple[str, str], ...] = ()
    independent_solution_evidence: bool = False

    @property
    def exact_literal_ids(self) -> frozenset[str]:
        return frozenset(
            item.literal_id for item in self.classifications if item.status is CompileStatus.EXACT
        )

    @property
    def candidate_keys(self) -> frozenset[str]:
        return frozenset(item.literal_id for item in self.candidate_audits)

    @property
    def witness_keys(self) -> frozenset[str]:
        return frozenset(item.literal_id for item in self.witness_audits)

    @property
    def deterministic(self) -> bool:
        required = int(load_intended_space()["compile_repetitions"])
        return len(self.problem_hashes) == required and len(set(self.problem_hashes)) == 1

    @property
    def evidence_complete(self) -> bool:
        return (
            self.candidate_keys == self.exact_literal_ids
            and self.witness_keys == self.exact_literal_ids
            and bool(self.coverage_cell)
            and self.independent_solution_evidence
        )

    @property
    def semantic_failures(self) -> int:
        candidate = sum(not audit.exact for audit in self.candidate_audits)
        witnesses = sum(not audit.passed for audit in self.witness_audits)
        return (
            candidate
            + witnesses
            + (not self.solution_set_audit.exact)
            + (not self.baseline_round_trip_ok)
            + (not self.deterministic)
            + self.silent_relaxations
        )


@dataclass(frozen=True, slots=True)
class Gate1ACensus:
    """Manifest-bound Gate report; assignment and audit counts are never caller supplied."""

    expected_assignment_ids: tuple[str, ...]
    audits: tuple[AssignmentAudit, ...]
    intended_literal_kinds: tuple[str, ...]
    meaningful_exact_tensor_subset: bool = False

    @classmethod
    def from_manifest(
        cls,
        audits: tuple[AssignmentAudit, ...],
        *,
        meaningful_exact_tensor_subset: bool = False,
    ) -> Gate1ACensus:
        manifest = load_intended_space()
        count = int(manifest["minimum_deterministic_assignments"])
        pattern = manifest["assignment_id_pattern"]
        expected = tuple(pattern.format(index=index) for index in range(count))
        kinds = tuple(manifest["operator_literals"]) + tuple(manifest["tensor_literals"]) + ("distribution_plan",)
        return cls(expected, audits, kinds, meaningful_exact_tensor_subset)

    @property
    def observed_assignment_ids(self) -> tuple[str, ...]:
        return tuple(audit.assignment_id for audit in self.audits)

    @property
    def evidence_complete(self) -> bool:
        manifest = load_intended_space()
        expected = set(self.expected_assignment_ids)
        observed = set(self.observed_assignment_ids)
        valid_dags = set(manifest["dag_classes"])
        required_cells = {
            (
                ("dag_class", dag),
                ("materialization", materialization),
                ("distribution", distribution),
                ("distribution_plan", plan),
            )
            for dag, materialization, distribution, plan in product(
                manifest["dag_classes"],
                manifest["tensor_literals"]["materialization"],
                manifest["tensor_literals"]["distribution"],
                manifest["distribution_plans"],
            )
        }
        observed_cells = {audit.coverage_cell for audit in self.audits}
        required_literal_kinds = set(self.intended_literal_kinds)
        return (
            len(self.observed_assignment_ids) == len(observed)
            and observed == expected
            and required_cells <= observed_cells
            and all(
                audit.dag_class in valid_dags
                and audit.evidence_complete
                and {
                    item.literal_id.rsplit(".", 1)[-1] for item in audit.classifications
                }
                == required_literal_kinds
                for audit in self.audits
            )
        )

    @property
    def false_exact(self) -> int:
        return sum(
            not audit.exact
            for assignment in self.audits
            for audit in assignment.candidate_audits
        )

    @property
    def coverage(self) -> float:
        represented = {
            classification.literal_id.rsplit(".", 1)[-1]
            for audit in self.audits
            for classification in audit.classifications
            if classification.status is CompileStatus.EXACT
        }
        return len(represented & set(self.intended_literal_kinds)) / len(self.intended_literal_kinds)

    @property
    def verdict(self) -> Gate1AVerdict:
        if not self.evidence_complete:
            return Gate1AVerdict.NOT_RUN
        if any(audit.semantic_failures for audit in self.audits):
            return Gate1AVerdict.FAIL
        if self.coverage == 1.0:
            return Gate1AVerdict.PASS
        return Gate1AVerdict.NARROW if self.meaningful_exact_tensor_subset else Gate1AVerdict.FAIL


@dataclass(frozen=True, slots=True)
class CandidateSetAudit:
    """Layer I exact set comparison for one literal."""

    literal_id: str
    intended: frozenset[str]
    compiled: frozenset[str]

    @property
    def sound(self) -> bool:
        return self.compiled <= self.intended

    @property
    def complete(self) -> bool:
        return self.intended <= self.compiled

    @property
    def exact(self) -> bool:
        return self.sound and self.complete


@dataclass(frozen=True, slots=True)
class ViolationWitnessAudit:
    """Layer II result: no compiled semantic solution may violate the literal."""

    literal_id: str
    violating_solutions: tuple[SemanticSolution, ...]

    @property
    def passed(self) -> bool:
        return not self.violating_solutions


@dataclass(frozen=True, slots=True)
class SolutionSetAudit:
    """Layer III equality after canonical semantic projection."""

    compiled: frozenset[SemanticSolution]
    reference: frozenset[SemanticSolution]

    @property
    def sound(self) -> bool:
        return self.compiled <= self.reference

    @property
    def complete(self) -> bool:
        return self.reference <= self.compiled

    @property
    def exact(self) -> bool:
        return self.compiled == self.reference


@dataclass(frozen=True, slots=True)
class Gate1AReport:
    """A Gate verdict whose missing evidence remains explicit as ``NOT_RUN``."""

    intended_literal_count: int
    classifications: tuple[LiteralClassification, ...]
    candidate_audits: tuple[CandidateSetAudit, ...] = ()
    witness_audits: tuple[ViolationWitnessAudit, ...] = ()
    solution_set_audits: tuple[SolutionSetAudit, ...] = ()
    baseline_round_trip_violations: int = 0
    nondeterministic_compiles: int = 0
    silent_relaxations: int = 0
    required_assignment_count: int = 1000
    audited_assignment_count: int = 0
    meaningful_exact_subset: bool = False

    @property
    def exact_literal_count(self) -> int:
        return sum(item.status is CompileStatus.EXACT for item in self.classifications)

    @property
    def coverage(self) -> float:
        return self.exact_literal_count / self.intended_literal_count if self.intended_literal_count else 0.0

    @property
    def false_exact(self) -> int:
        audit_by_literal = {audit.literal_id: audit for audit in self.candidate_audits}
        return sum(
            item.status is CompileStatus.EXACT
            and (item.literal_id not in audit_by_literal or not audit_by_literal[item.literal_id].exact)
            for item in self.classifications
        )

    @property
    def verdict(self) -> Gate1AVerdict:
        evidence_complete = (
            self.audited_assignment_count >= self.required_assignment_count
            and len(self.witness_audits) == self.exact_literal_count
            and bool(self.solution_set_audits)
        )
        if not evidence_complete:
            return Gate1AVerdict.NOT_RUN
        semantic_failures = (
            self.false_exact
            + self.silent_relaxations
            + self.baseline_round_trip_violations
            + self.nondeterministic_compiles
            + sum(not audit.passed for audit in self.witness_audits)
            + sum(not audit.exact for audit in self.solution_set_audits)
        )
        if semantic_failures:
            return Gate1AVerdict.FAIL
        if self.coverage == 1.0:
            return Gate1AVerdict.PASS
        return Gate1AVerdict.NARROW if self.meaningful_exact_subset else Gate1AVerdict.FAIL


def audit_candidate_set(
    reference: CompiledStructuralAssignment,
    compiled: CompiledStructuralAssignment,
    literal: StructuralLiteral,
) -> CandidateSetAudit:
    """Compare ``Options_0(target) intersect allowed`` with compiled options."""

    reference_options = _options(reference, literal, independent=True)
    intended = frozenset(reference_options & set(literal.allowed))
    compiled_options = frozenset(_options(compiled, literal))
    return CandidateSetAudit(literal.literal_id, intended, compiled_options)


def audit_determinism(compilations: tuple[CompiledStructuralAssignment, ...]) -> bool:
    """Require identical semantic hashes across repeated compilations."""

    return bool(compilations) and len({result.semantic_hash for result in compilations}) == 1


def audit_baseline_round_trip(
    baseline: CompiledStructuralAssignment,
    round_trip: CompiledStructuralAssignment,
) -> bool:
    """Compare semantic manifests; backend serialization is intentionally diagnostic only."""

    return (
        baseline.stage.value == "post_transfer"
        and round_trip.stage.value == "post_transfer"
        and baseline.pipeline_manifest is not None
        and round_trip.pipeline_manifest is not None
        and baseline.problem_hash == round_trip.problem_hash
    )


def audit_violation_witness(
    literal: StructuralLiteral,
    compiled_solutions: frozenset[SemanticSolution],
) -> ViolationWitnessAudit:
    """Finite witness query equivalent to ``C(x) and not literal`` for micro problems."""

    violations = tuple(
        solution
        for solution in sorted(compiled_solutions)
        if dict(solution).get(literal.literal_id) not in literal.allowed
    )
    return ViolationWitnessAudit(literal.literal_id, violations)


def audit_solution_sets(
    compiled: frozenset[SemanticSolution],
    reference: frozenset[SemanticSolution],
) -> SolutionSetAudit:
    return SolutionSetAudit(compiled=compiled, reference=reference)


def enumerate_tta_semantic_solutions(
    tta: TransferAndTensorAllocator,
    *,
    fixed_literals: tuple[tuple[str, str], ...] = (),
    max_solutions: int = 10_000,
) -> frozenset[SemanticSolution]:
    """Enumerate independent primary-decision projections with solver no-good cuts."""

    solutions: set[SemanticSolution] = set()
    while len(solutions) < max_solutions:
        tta.model.optimize()
        status = tta.model.get_status()
        if status == "INFEASIBLE":
            return frozenset(solutions)
        if status != "OPTIMAL":
            raise RuntimeError(f"feasible-set enumeration ended with {status}")
        selected = []
        projection = list(fixed_literals)
        for (tensor, choice), variable in tta.x_tensor_choice.items():
            if variable.X > tta.VAR_THRESHOLD:
                selected.append(variable)
                projection.append((f"tensor:{tensor.name}", _reference_core_choice(choice)))
        for (transfer, choice), variable in tta.y_path_choice.items():
            if variable.X > tta.VAR_THRESHOLD:
                selected.append(variable)
                projection.append((f"path:{transfer.name}", reference_path_key(choice)))
        for (tensor, stop), variable in tta.z_stop.items():
            if variable.X > tta.VAR_THRESHOLD:
                selected.append(variable)
                projection.append((f"reuse:{tensor.name}", str(stop)))
        semantic = tuple(sorted(projection))
        if semantic in solutions:
            raise RuntimeError("solver returned a duplicate semantic projection after a no-good cut")
        solutions.add(semantic)
        if not selected:
            return frozenset(solutions)
        tta.model.add_constr(
            tta.model.quicksum(variable._raw for variable in selected) <= len(selected) - 1,
            name=f"gate1a_no_good_{len(solutions)}",
        )
    raise RuntimeError(f"feasible-set enumeration exceeded {max_solutions} solutions")


def _reference_core_choice(choice) -> str:
    return "cores:" + ",".join(str(core.id) for core in choice)


def _options(
    compilation: CompiledStructuralAssignment,
    literal: StructuralLiteral,
    *,
    independent: bool = False,
) -> set[str]:
    mapping = compilation.mapping
    nodes = [node for node in mapping.nodes() if node.name == literal.target]
    if len(nodes) != 1:
        raise ValueError(f"target {literal.target!r} is not unique")
    node = nodes[0]
    node_mapping = mapping.get(node)
    if literal.kind.value == "hardware_zone" and isinstance(node, ComputationNode):
        return {core_group_key(tuple(option)) for option in node_mapping.resource_allocation}
    if literal.kind.value == "inter_core_tiling" and isinstance(node, ComputationNode):
        return {tiling_key(option) for option in node_mapping.inter_core_tiling}
    if literal.kind.value == "transfer_path" and isinstance(node, TransferNode):
        project = reference_path_key if independent else path_key
        return {project(option) for option in node_mapping.resource_allocation}
    raise ValueError(f"literal {literal.literal_id!r} has no candidate-set projection")


def reference_path_key(path) -> str:
    """Independent Layer-I projection for a solver-visible multicast path.

    Deliberately does not call the compiler's ``path_key``: agreement between
    the compiler and this projection therefore detects omitted path fields.
    """

    import json  # noqa: PLC0415

    return json.dumps(
        {
            "sources": [_reference_resource_id(core) for core in path.sources],
            "targets": [_reference_resource_id(core) for core in path.targets],
            "total_hops_objective": path.total_hops_objective,
            "links": [
                (_reference_resource_id(link.sender), _reference_resource_id(link.receiver))
                for link in path.links_used
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _reference_resource_id(resource) -> str:
    if type(resource) is str:
        return "str:" + resource
    identifier = getattr(resource, "id", None)
    if identifier is None:
        raise TypeError("reference projection received an endpoint without id")
    return type(resource).__name__ + ":" + str(identifier)


def mapping_semantics(compilation: CompiledStructuralAssignment) -> dict:
    """Public helper for conformance artifacts."""

    return canonical_mapping_manifest(compilation.mapping)
