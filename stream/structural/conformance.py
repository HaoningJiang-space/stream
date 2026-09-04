"""Evidence records for the three Gate 1A conformance layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from stream.structural.stream_contract import (
    CompiledStructuralAssignment,
    CompileStatus,
    LiteralClassification,
    StructuralLiteral,
    canonical_mapping_manifest,
    core_group_key,
    path_key,
    tiling_key,
)
from stream.workload.node import ComputationNode, TransferNode

SemanticSolution: TypeAlias = tuple[tuple[str, str], ...]


class Gate1AVerdict(StrEnum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    NARROW = "NARROW"
    FAIL = "FAIL"


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

    reference_options = _options(reference, literal)
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
        and baseline.semantic_hash == round_trip.semantic_hash
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


def _options(compilation: CompiledStructuralAssignment, literal: StructuralLiteral) -> set[str]:
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
        return {path_key(option) for option in node_mapping.resource_allocation}
    raise ValueError(f"literal {literal.literal_id!r} has no candidate-set projection")


def mapping_semantics(compilation: CompiledStructuralAssignment) -> dict:
    """Public helper for conformance artifacts."""

    return canonical_mapping_manifest(compilation.mapping)
