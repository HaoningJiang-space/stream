"""Executable Gate 1A-v2 audit for the minimal tensor restriction interface."""

from __future__ import annotations

import json
import platform
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from itertools import cycle, islice, product
from pathlib import Path
from typing import Any

from stream.opt.allocation.constraint_optimization.tensor_restriction import (
    TensorRestriction,
    TransferPlanRestriction,
    tensor_placement_key,
    transfer_plan_key,
)
from stream.structural.conformance import enumerate_tta_semantic_solutions, reference_path_key
from stream.structural.gate1a_cases import build_tensor_restriction_scheduler
from stream.structural.pipeline import (
    PreparedTensorRestrictedProblem,
    prepare_tensor_restricted_problem,
    prepare_uninstrumented_reference_problem,
)
from stream.structural.stream_contract import Gate1AEvalConfig
from stream.workload.node import TransferNode
from stream.workload.tensor import Tensor

_CONSTRAINTS = ("memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels", "pipelining")
_MIN_STAGING_TRANSFERS = 2
_HISTORICAL_V1_ARTIFACT = Path("artifacts/gate1a/report.json")


@dataclass(frozen=True, slots=True)
class ReferenceEvidence:
    target_tensor: Tensor
    adjacent_transfers: tuple[TransferNode, ...]
    placement_options: tuple
    path_options: tuple[tuple[TransferNode, tuple], ...]
    solutions: frozenset[tuple[tuple[str, str], ...]]
    baseline_round_trip_ok: bool
    baseline_problem_hash: str


def run_tensor_restriction_gate(output_path: str | Path) -> dict[str, Any]:
    """Run the frozen Gate 1A-v2 census and write a replayable JSON artifact."""

    manifest = _load_manifest()
    historical_v1 = _historical_v1_manifest()
    eval_config = Gate1AEvalConfig(backend=manifest["solver"]["backend"], constraints=_CONSTRAINTS)
    references = {
        dag_class: _reference_evidence(dag_class, eval_config, manifest) for dag_class in manifest["dag_classes"]
    }
    cells = tuple(product(manifest["dag_classes"], range(len(manifest["allowed_placement_index_sets"]))))
    cell_evidence = {
        _cell_id(dag_class, subset_index): _audit_cell(
            dag_class,
            subset_index,
            references[dag_class],
            eval_config,
            manifest,
        )
        for dag_class, subset_index in cells
    }

    count = int(manifest["minimum_deterministic_assignments"])
    repetitions = int(manifest["compile_repetitions"])
    assignment_ids = [manifest["assignment_id_pattern"].format(index=index) for index in range(count)]
    scheduled_cells = tuple(islice(cycle(cells), count))
    proofs = []
    for assignment_id, (dag_class, subset_index) in zip(assignment_ids, scheduled_cells, strict=True):
        reference = references[dag_class]
        restriction, _ = _restriction_for_subset(reference, subset_index, manifest)
        hashes = tuple(_compile_domain_hash(dag_class, restriction, eval_config) for _ in range(repetitions))
        proofs.append(
            {
                "assignment_id": assignment_id,
                "cell": _cell_id(dag_class, subset_index),
                "problem_hashes": hashes,
                "deterministic": len(set(hashes)) == 1,
            }
        )

    required_cells = {_cell_id(dag_class, subset_index) for dag_class, subset_index in cells}
    observed_cells = {proof["cell"] for proof in proofs}
    semantic_failures = sum(not evidence["exact"] for evidence in cell_evidence.values())
    false_exact = sum(evidence["false_exact"] for evidence in cell_evidence.values())
    silent_relaxations = sum(evidence["silent_relaxations"] for evidence in cell_evidence.values())
    unintended_restrictions = sum(evidence["unintended_restrictions"] for evidence in cell_evidence.values())
    nondeterministic_compiles = sum(not proof["deterministic"] for proof in proofs)
    baseline_round_trip_failures = sum(not evidence.baseline_round_trip_ok for evidence in references.values())
    nontrivial = any(evidence["nontrivial"] for evidence in cell_evidence.values())
    evidence_complete = (
        len(proofs) == count
        and len({proof["assignment_id"] for proof in proofs}) == count
        and required_cells <= observed_cells
        and all(len(proof["problem_hashes"]) == repetitions for proof in proofs)
        and all(evidence["evidence_complete"] for evidence in cell_evidence.values())
    )
    coverage = 1.0 if evidence_complete else 0.0
    passed = (
        evidence_complete
        and nontrivial
        and false_exact == 0
        and silent_relaxations == 0
        and unintended_restrictions == 0
        and semantic_failures == 0
        and nondeterministic_compiles == 0
        and baseline_round_trip_failures == 0
    )
    payload = {
        "contract": manifest["contract"],
        "version": manifest["version"],
        "verdict": "PASS" if passed else "FAIL",
        "coverage": coverage,
        "evidence_complete": evidence_complete,
        "nontrivial_tensor_space": nontrivial,
        "assignment_count": len(proofs),
        "coverage_cell_count": len(observed_cells),
        "required_coverage_cell_count": len(required_cells),
        "false_exact": false_exact,
        "silent_relaxations": silent_relaxations,
        "unintended_restrictions": unintended_restrictions,
        "semantic_failures": semantic_failures,
        "nondeterministic_compiles": nondeterministic_compiles,
        "baseline_round_trip_failures": baseline_round_trip_failures,
        "scope": {
            "exact": manifest["exact_tensor_literals"],
            "unsupported": manifest["unsupported_literals"],
            "claim": "exact finite restriction of existing TTA placement and path domains",
        },
        "evidence_basis": {
            "candidate_sets": "independent raw-option projection against selected reference choices",
            "soundness": "all restricted semantic solutions are in unrestricted baseline intersect Gamma",
            "completeness": "all unrestricted baseline solutions satisfying Gamma remain restricted-feasible",
            "oracle_independence": (
                "fresh unrestricted scheduler plus independent core/path projection; restriction helpers are not used"
            ),
        },
        "cells": cell_evidence,
        "references": {
            dag_class: {
                "target_tensor": evidence.target_tensor.name,
                "adjacent_transfers": [transfer.name for transfer in evidence.adjacent_transfers],
                "baseline_placement_option_count": len(evidence.placement_options),
                "baseline_path_option_counts": {
                    transfer.name: len(options) for transfer, options in evidence.path_options
                },
                "baseline_solution_count": len(evidence.solutions),
                "baseline_round_trip_ok": evidence.baseline_round_trip_ok,
                "baseline_problem_hash": evidence.baseline_problem_hash,
            }
            for dag_class, evidence in references.items()
        },
        "assignment_proofs": proofs,
        "environment": _environment_manifest(),
        "historical_v1": historical_v1,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _reference_evidence(
    dag_class: str,
    eval_config: Gate1AEvalConfig,
    manifest: dict[str, Any],
) -> ReferenceEvidence:
    direct = prepare_uninstrumented_reference_problem(
        build_tensor_restriction_scheduler(dag_class),
        eval_config,
    )
    empty = prepare_tensor_restricted_problem(
        build_tensor_restriction_scheduler(dag_class),
        (),
        eval_config,
    )
    direct_tta = direct.build_tta()
    empty_tta = empty.build_tta()
    target, adjacent = _select_target(direct_tta)
    placement_options = direct_tta.possible_tensor_allocations[target]
    path_options = tuple((transfer, direct_tta.possible_transfer_allocations[transfer]) for transfer in adjacent)
    if len(placement_options) < manifest["minimum_baseline_placement_options"]:
        raise RuntimeError(f"{dag_class}: placement domain is not preregistered-nontrivial")
    if any(len(options) < manifest["minimum_baseline_path_options"] for _, options in path_options):
        raise RuntimeError(f"{dag_class}: path domain is not preregistered-nontrivial")
    direct_solutions = enumerate_tta_semantic_solutions(
        direct_tta,
        max_solutions=manifest["solver"]["enumeration_limit"],
    )
    empty_solutions = enumerate_tta_semantic_solutions(
        empty_tta,
        max_solutions=manifest["solver"]["enumeration_limit"],
    )
    round_trip = (
        direct.compilation.pipeline_manifest == empty.pipeline_manifest
        and _domain_manifest(direct_tta) == _domain_manifest(empty_tta)
        and direct_solutions == empty_solutions
        and not empty.restrictions
    )
    return ReferenceEvidence(
        target,
        adjacent,
        placement_options,
        path_options,
        direct_solutions,
        round_trip,
        direct.compilation.problem_hash,
    )


def _audit_cell(
    dag_class: str,
    subset_index: int,
    reference: ReferenceEvidence,
    eval_config: Gate1AEvalConfig,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    restriction, intended = _restriction_for_subset(reference, subset_index, manifest)
    prepared = prepare_tensor_restricted_problem(
        build_tensor_restriction_scheduler(dag_class),
        restriction,
        eval_config,
    )
    tta = prepared.build_tta()
    candidate_domains = _selected_domain_manifest(tta, reference.target_tensor.name, reference.adjacent_transfers)
    candidate_sets_exact = candidate_domains == intended
    candidate_solutions = enumerate_tta_semantic_solutions(
        tta,
        max_solutions=manifest["solver"]["enumeration_limit"],
    )
    expected_solutions = frozenset(
        solution for solution in reference.solutions if _independent_solution_satisfies(solution, intended)
    )
    sound = candidate_solutions <= expected_solutions
    complete = expected_solutions <= candidate_solutions
    every_allowed_witnessed = all(
        any(dict(solution).get(variable) == choice for solution in expected_solutions)
        for variable, choices in intended.items()
        for choice in choices
    )
    return {
        "dag_class": dag_class,
        "placement_index_set": manifest["allowed_placement_index_sets"][subset_index],
        "restriction_manifest": prepared.restriction_manifest,
        "baseline_solution_count": len(reference.solutions),
        "expected_solution_count": len(expected_solutions),
        "compiled_solution_count": len(candidate_solutions),
        "candidate_sets_exact": candidate_sets_exact,
        "sound": sound,
        "complete": complete,
        "every_allowed_choice_witnessed": every_allowed_witnessed,
        "false_exact": int(not candidate_sets_exact),
        "silent_relaxations": int(not sound),
        "unintended_restrictions": int(not complete),
        "nontrivial": any(len(choices) > 1 for choices in intended.values()),
        "evidence_complete": bool(expected_solutions) and every_allowed_witnessed,
        "exact": candidate_sets_exact and sound and complete and every_allowed_witnessed,
        "problem_hash": _compiled_domain_hash(prepared, tta),
    }


def _restriction_for_subset(
    reference: ReferenceEvidence,
    subset_index: int,
    manifest: dict[str, Any],
) -> tuple[tuple[TensorRestriction, ...], dict[str, frozenset[str]]]:
    indices = tuple(manifest["allowed_placement_index_sets"][subset_index])
    selected_placements = tuple(reference.placement_options[index] for index in indices)
    production_placement_keys = frozenset(tensor_placement_key(option) for option in selected_placements)
    intended: dict[str, frozenset[str]] = {
        f"tensor:{reference.target_tensor.name}": frozenset(_independent_placement_key(x) for x in selected_placements)
    }
    transfer_restrictions = []
    for transfer, options in reference.path_options:
        selected_paths = tuple(
            path
            for path in options
            if _path_endpoint_placement_key(path, transfer, reference.target_tensor) in production_placement_keys
        )
        if not selected_paths:
            raise RuntimeError(f"{transfer.name}: no path realizes placement subset {indices}")
        transfer_restrictions.append(
            TransferPlanRestriction(
                transfer.name,
                frozenset(transfer_plan_key(path) for path in selected_paths),
            )
        )
        intended[f"path:{transfer.name}"] = frozenset(reference_path_key(path) for path in selected_paths)
    return (
        (
            TensorRestriction(
                reference.target_tensor.name,
                production_placement_keys,
                tuple(transfer_restrictions),
            ),
        ),
        intended,
    )


def _select_target(tta) -> tuple[Tensor, tuple[TransferNode, ...]]:
    for tensor, placements in tta.possible_tensor_allocations.items():
        if len(placements) <= 1:
            continue
        producers = tuple(transfer for transfer in tta.transfer_nodes if tensor in transfer.outputs)
        adjacent = tuple(transfer for transfer in tta.transfer_nodes if tensor in transfer.tensors)
        if len(producers) == 1 and len(adjacent) >= _MIN_STAGING_TRANSFERS:
            return tensor, adjacent
    raise RuntimeError("no nontrivial transfer-produced staging tensor exists")


def _path_endpoint_placement_key(path, transfer: TransferNode, tensor: Tensor) -> str:
    if tensor in transfer.inputs:
        return tensor_placement_key(tuple(path.sources))
    if tensor in transfer.outputs:
        return tensor_placement_key(tuple(path.targets))
    raise RuntimeError(f"{transfer.name} is not adjacent to {tensor.name}")


def _selected_domain_manifest(tta, tensor_name: str, transfers: tuple[TransferNode, ...]) -> dict[str, frozenset[str]]:
    tensor = next(tensor for tensor in tta.possible_tensor_allocations if tensor.name == tensor_name)
    transfer_by_name = {transfer.name: transfer for transfer in tta.possible_transfer_allocations}
    return {
        f"tensor:{tensor_name}": frozenset(
            _independent_placement_key(option) for option in tta.possible_tensor_allocations[tensor]
        ),
        **{
            f"path:{reference_transfer.name}": frozenset(
                reference_path_key(option)
                for option in tta.possible_transfer_allocations[transfer_by_name[reference_transfer.name]]
            )
            for reference_transfer in transfers
        },
    }


def _independent_solution_satisfies(
    solution: tuple[tuple[str, str], ...],
    intended: dict[str, frozenset[str]],
) -> bool:
    observed = dict(solution)
    return all(observed.get(variable) in choices for variable, choices in intended.items())


def _compile_domain_hash(
    dag_class: str,
    restriction: tuple[TensorRestriction, ...],
    eval_config: Gate1AEvalConfig,
) -> str:
    prepared = prepare_tensor_restricted_problem(
        build_tensor_restriction_scheduler(dag_class),
        restriction,
        eval_config,
    )
    return _compiled_domain_hash(prepared, prepared.build_tta())


def _compiled_domain_hash(prepared: PreparedTensorRestrictedProblem, tta) -> str:
    return _digest(
        {
            "problem": prepared.problem_manifest,
            "tta_domains": _domain_manifest(tta),
        }
    )


def _domain_manifest(tta) -> dict[str, Any]:
    return {
        "tensors": {
            tensor.name: [_independent_placement_key(option) for option in options]
            for tensor, options in sorted(tta.possible_tensor_allocations.items(), key=lambda item: item[0].name)
        },
        "transfers": {
            transfer.name: [reference_path_key(option) for option in options]
            for transfer, options in sorted(tta.possible_transfer_allocations.items(), key=lambda item: item[0].name)
        },
    }


def _independent_placement_key(choice) -> str:
    return "cores:" + ",".join(str(core.id) for core in choice)


def _cell_id(dag_class: str, subset_index: int) -> str:
    return f"{dag_class}:placement-subset-{subset_index}"


def _load_manifest() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath("gate1a_intended_space_v2.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _historical_v1_manifest() -> dict[str, str]:
    if not _HISTORICAL_V1_ARTIFACT.is_file():
        raise FileNotFoundError(
            f"historical Gate 1A-v1 artifact is required before running v2: {_HISTORICAL_V1_ARTIFACT}"
        )
    report = json.loads(_HISTORICAL_V1_ARTIFACT.read_text(encoding="utf-8"))
    if report.get("verdict") != "FAIL":
        raise RuntimeError("historical Gate 1A-v1 artifact no longer records the frozen FAIL result")
    return {
        "commit": "0b7d3589f1605bdf047f4e9b5dba67da176de10b",
        "artifact": str(_HISTORICAL_V1_ARTIFACT),
        "artifact_sha256": sha256(_HISTORICAL_V1_ARTIFACT.read_bytes()).hexdigest(),
        "verdict": "FAIL",
    }


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _environment_manifest() -> dict[str, Any]:
    source_paths = (
        "stream/cost_model/steady_state_scheduler.py",
        "stream/opt/allocation/constraint_optimization/tensor_restriction.py",
        "stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py",
        "stream/structural/conformance.py",
        "stream/structural/gate1a_cases.py",
        "stream/structural/pipeline.py",
        "stream/structural/tensor_restriction_gate.py",
        "stream/structural/contracts/gate1a_intended_space_v2.json",
        "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml",
        "pyproject.toml",
    )
    packages = {}
    for package in ("ortools", "stream-dse", "xdsl", "zigzag-dse"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "completion_marker": "COMPLETE",
        "command": "python scripts/run_tensor_restriction_gate.py --output <path>",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "packages": packages,
        "declared_dependencies": pyproject["project"]["dependencies"],
        "source_sha256": {path: sha256(Path(path).read_bytes()).hexdigest() for path in source_paths},
    }
