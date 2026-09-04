"""Executable, manifest-bound Gate 1A conformance census."""

from __future__ import annotations

import json
import platform
import sys
import tomllib
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path

from stream.structural.conformance import (
    AssignmentAudit,
    CandidateSetAudit,
    Gate1ACensus,
    audit_baseline_round_trip,
    audit_solution_sets,
    audit_violation_witness,
    enumerate_tta_semantic_solutions,
)
from stream.structural.gate1a_cases import build_gate1a_scheduler
from stream.structural.pipeline import (
    prepare_reference_problem,
    prepare_structural_problem,
    prepare_uninstrumented_reference_problem,
)
from stream.structural.stream_contract import (
    CompileStatus,
    Gate1AEvalConfig,
    LiteralKind,
    StructuralLiteral,
    StructuralMappingContract,
    canonical_mapping_manifest,
    core_group_key,
    load_intended_space,
    tiling_key,
)
from stream.workload.node import ComputationNode

_CONSTRAINTS = ("memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels", "pipelining")


def run_gate1a(output_path: str | Path) -> Gate1ACensus:
    """Run six production-pipeline equivalence classes and expand them to the frozen 1,000 assignments."""

    manifest = load_intended_space()
    eval_config = Gate1AEvalConfig(backend="ORTOOLS_GSCIP", constraints=_CONSTRAINTS)
    class_evidence = {
        dag_class: _audit_equivalence_class(dag_class, eval_config) for dag_class in manifest["dag_classes"]
    }
    axes = tuple(
        product(
            manifest["dag_classes"],
            manifest["tensor_literals"]["materialization"],
            manifest["tensor_literals"]["distribution"],
            manifest["distribution_plans"],
        )
    )
    assignments = []
    assignment_proofs = []
    count = int(manifest["minimum_deterministic_assignments"])
    pattern = manifest["assignment_id_pattern"]
    for index in range(count):
        dag_class, materialization, distribution, distribution_plan = axes[index % len(axes)]
        evidence = class_evidence[dag_class]
        cell = _audit_tensor_cell(
            dag_class,
            materialization,
            distribution,
            distribution_plan,
            evidence,
            eval_config,
        )
        assignment_id = pattern.format(index=index)
        assignments.append(
            AssignmentAudit(
                dag_class=dag_class,
                assignment_id=assignment_id,
                classifications=cell["classifications"],
                candidate_audits=evidence["candidate_audits"],
                witness_audits=evidence["witness_audits"],
                solution_set_audit=evidence["solution_set_audit"],
                problem_hashes=cell["problem_hashes"],
                baseline_round_trip_ok=evidence["baseline_round_trip_ok"],
                coverage_cell=(
                    ("dag_class", dag_class),
                    ("materialization", materialization),
                    ("distribution", distribution),
                    ("distribution_plan", distribution_plan),
                ),
                independent_solution_evidence=evidence["independent_solution_evidence"],
            )
        )
        assignment_proofs.append(
            {
                "assignment_id": assignment_id,
                "dag_class": dag_class,
                "materialization": materialization,
                "distribution": distribution,
                "distribution_plan": distribution_plan,
                "problem_hashes": cell["problem_hashes"],
            }
        )
    tensor_kinds = {"materialization", "distribution", "distribution_plan"}
    meaningful_tensor_subset = any(
        item.status is CompileStatus.EXACT and item.literal_id.rsplit(".", 1)[-1] in tensor_kinds
        for audit in assignments
        for item in audit.classifications
    )
    census = Gate1ACensus.from_manifest(
        tuple(assignments), meaningful_exact_tensor_subset=meaningful_tensor_subset
    )
    _write_artifact(Path(output_path), manifest, census, class_evidence, assignment_proofs)
    return census


def _audit_tensor_cell(
    dag_class: str,
    materialization: str,
    distribution: str,
    distribution_plan: str,
    class_evidence: dict,
    eval_config: Gate1AEvalConfig,
) -> dict:
    scheduler = build_gate1a_scheduler(dag_class)
    tensor_name = next(iter(scheduler.workload.get_computation_nodes())).outputs[0].name
    tensor_literals = (
        StructuralLiteral("tensor.materialization", LiteralKind.MATERIALIZATION, tensor_name, (materialization,)),
        StructuralLiteral("tensor.distribution", LiteralKind.DISTRIBUTION, tensor_name, (distribution,)),
        StructuralLiteral(
            "tensor.distribution_plan",
            LiteralKind.DISTRIBUTION_PLAN,
            tensor_name,
            (distribution_plan,),
        ),
    )
    contract = StructuralMappingContract(
        f"{dag_class}-{materialization}-{distribution}-{distribution_plan}",
        class_evidence["operator_literals"] + tensor_literals,
    )
    compilations = tuple(
        prepare_structural_problem(build_gate1a_scheduler(dag_class), contract, eval_config).compilation
        for _ in range(3)
    )
    hashes = tuple(compilation.problem_hash for compilation in compilations)
    if set(hashes) != {class_evidence["problem_hashes"][0]}:
        raise RuntimeError(f"unsupported tensor literals mutated the {dag_class} problem")
    return {"classifications": compilations[0].classifications, "problem_hashes": hashes}


def _audit_equivalence_class(dag_class: str, eval_config: Gate1AEvalConfig) -> dict:
    references = (
        prepare_uninstrumented_reference_problem(build_gate1a_scheduler(dag_class), eval_config),
        prepare_reference_problem(build_gate1a_scheduler(dag_class), eval_config),
    )
    source_scheduler = build_gate1a_scheduler(dag_class)
    node = next(iter(source_scheduler.workload.get_computation_nodes()))
    node_mapping = source_scheduler.mapping.get(node)
    literals = (
        StructuralLiteral(
            "operator.hardware_zone",
            LiteralKind.HARDWARE_ZONE,
            node.name,
            (core_group_key(tuple(node_mapping.resource_allocation[0])),),
        ),
        StructuralLiteral(
            "operator.inter_core_tiling",
            LiteralKind.OPERATOR_TILING,
            node.name,
            (tiling_key(node_mapping.inter_core_tiling[0]),),
        ),
    )
    contract = StructuralMappingContract(f"{dag_class}-operator-class", literals)
    candidates = tuple(
        prepare_structural_problem(build_gate1a_scheduler(dag_class), contract, eval_config) for _ in range(3)
    )
    first = candidates[0]
    candidate_tta = first.build_tta()
    reference_tta = references[0].build_tta()
    if candidate_tta.mapping is not first.compilation.mapping or candidate_tta.slot_of is not first.timeslots:
        raise RuntimeError(f"{dag_class}: production TTA handoff reconstructed restricted objects")
    exact = tuple(item for item in first.compilation.classifications if item.status is CompileStatus.EXACT)
    if len(exact) != len(literals):
        raise RuntimeError(f"{dag_class}: preregistered operator literals were not EXACT")
    reference_node = _node_by_name(references[0].compilation, node.name)
    candidate_node = _node_by_name(first.compilation, node.name)
    ref_mapping = references[0].compilation.mapping.get(reference_node)
    candidate_mapping = first.compilation.mapping.get(candidate_node)
    candidate_audits = (
        CandidateSetAudit(
            literals[0].literal_id,
            frozenset(_reference_core_group_key(option) for option in ref_mapping.resource_allocation),
            frozenset(_reference_core_group_key(option) for option in candidate_mapping.resource_allocation),
        ),
        CandidateSetAudit(
            literals[1].literal_id,
            frozenset(_reference_tiling_key(option) for option in ref_mapping.inter_core_tiling),
            frozenset(_reference_tiling_key(option) for option in candidate_mapping.inter_core_tiling),
        ),
    )
    fixed_literals = tuple((literal.literal_id, literal.allowed[0]) for literal in literals)
    reference_solutions = enumerate_tta_semantic_solutions(reference_tta, fixed_literals=fixed_literals)
    candidate_solutions = enumerate_tta_semantic_solutions(candidate_tta, fixed_literals=fixed_literals)
    witness_audits = tuple(audit_violation_witness(literal, candidate_solutions) for literal in literals)
    return {
        "classifications": first.compilation.classifications,
        "operator_literals": literals,
        "candidate_audits": candidate_audits,
        "witness_audits": witness_audits,
        "solution_set_audit": audit_solution_sets(candidate_solutions, reference_solutions),
        "independent_solution_evidence": True,
        "problem_hashes": tuple(candidate.compilation.problem_hash for candidate in candidates),
        "baseline_round_trip_ok": audit_baseline_round_trip(references[0].compilation, references[1].compilation),
        "reference_problem_hash": references[0].compilation.problem_hash,
        "mapping_manifest": canonical_mapping_manifest(first.compilation.mapping),
        "tensor_option_domain_count": len(candidate_tta.possible_tensor_allocations),
        "transfer_option_domain_count": len(candidate_tta.possible_transfer_allocations),
        "enumerated_solution_count": len(candidate_solutions),
    }


def _node_by_name(compilation, name: str) -> ComputationNode:
    nodes = [
        node
        for node in compilation.mapping.nodes()
        if isinstance(node, ComputationNode) and node.name == name
    ]
    if len(nodes) != 1:
        raise RuntimeError(f"expected one computation node named {name!r}")
    return nodes[0]


def _reference_core_group_key(group) -> str:
    return "cores:" + ",".join(str(core.id) for core in group)


def _reference_tiling_key(tiling) -> str:
    return "tiling:" + ",".join(str(dim) + "=" + str(factor) for dim, factor in tiling)


def _write_artifact(
    path: Path,
    manifest: dict,
    census: Gate1ACensus,
    class_evidence: dict,
    assignment_proofs: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": manifest["contract"],
        "version": manifest["version"],
        "verdict": census.verdict.value,
        "coverage": census.coverage,
        "evidence_complete": census.evidence_complete,
        "assignment_count": len(census.audits),
        "false_exact": census.false_exact,
        "semantic_failures": sum(audit.semantic_failures for audit in census.audits),
        "meaningful_exact_tensor_subset": census.meaningful_exact_tensor_subset,
        "evidence_basis": {
            "layer_i": "independent candidate-set equality at the final production mapping",
            "layer_ii": "solver-enumerated compiled projections checked for literal violations",
            "layer_iii": "independent plain and compiled TTA primary-decision feasible-set enumeration",
            "equivalence_reuse": (
                "1000 assignments reduce exactly to six DAG classes because unsupported literals "
                "do not mutate the problem"
            ),
        },
        "assignment_proofs": assignment_proofs,
        "environment": _environment_manifest(),
        "equivalence_classes": {
            dag_class: {
                key: value
                for key, value in evidence.items()
                if key
                in {
                    "problem_hashes",
                    "baseline_round_trip_ok",
                    "reference_problem_hash",
                    "tensor_option_domain_count",
                    "transfer_option_domain_count",
                    "enumerated_solution_count",
                }
            }
            for dag_class, evidence in class_evidence.items()
        },
        "classifications": [
            {
                "literal_id": item.literal_id,
                "status": item.status.value,
                "stage": item.stage.value if item.stage else None,
                "reason": item.reason.value if item.reason else None,
            }
            for item in census.audits[0].classifications
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _environment_manifest() -> dict:
    source_paths = (
        "stream/cost_model/steady_state_scheduler.py",
        "stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py",
        "stream/structural/stream_contract.py",
        "stream/structural/pipeline.py",
        "stream/structural/gate1a.py",
        "stream/structural/gate1a_cases.py",
        "stream/structural/conformance.py",
        "stream/structural/contracts/gate1a_intended_space_v1.json",
        "stream/opt/solver/solver.py",
        "stream/opt/allocation/constraint_optimization/context.py",
        "stream/inputs/examples/hardware/tpu_like_quad_core.yaml",
        "pyproject.toml",
    )
    packages = {}
    for package in ("ortools", "stream-dse", "xdsl", "zigzag-dse"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    ortools_requirement = next(
        dependency for dependency in pyproject["project"]["dependencies"] if dependency.startswith("ortools")
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": "python scripts/run_structural_gate1a.py --output <path>",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "packages": packages,
        "declared_ortools_requirement": ortools_requirement.removeprefix("ortools"),
        "source_sha256": {path: sha256(Path(path).read_bytes()).hexdigest() for path in source_paths},
    }
