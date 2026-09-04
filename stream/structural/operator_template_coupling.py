"""Gate 1A-v4: exact shared-tensor operator-template compatibility census."""

from __future__ import annotations

import json
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from stream.structural.gate1a_cases import build_gate1a_workload, load_gate1a_accelerator
from stream.structural.operator_template_contract import OperatorTemplateAssignment
from stream.structural.operator_template_faithfulness import (
    _baseline_mapping,
    _direct_reference_mapping_for_templates,
    _environment_manifest,
    _expected_pair,
    _file_digest,
    _prepared_problem_outcome,
    _run_compiler_only,
    _run_tiling_pipeline,
    _selected_pair,
    _source_manifest,
    _source_run_manifest,
    _static_review_manifest,
    _template_from_spec,
    _template_library,
)
from stream.structural.stream_contract import canonical_mapping_manifest

_CONTRACT_VERSION = 4
_RUNNER = Path("scripts/run_operator_template_coupling.py")


class OperatorTemplateCouplingError(RuntimeError):
    """The frozen Gate 1A-v4 contract or evidence is incomplete."""


def load_operator_template_coupling_contract() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath("gate1a_operator_template_v4.json")
    contract = json.loads(resource.read_text(encoding="utf-8"))
    _validate_contract(contract)
    return contract


def run_operator_template_coupling(
    output_path: str | Path,
    *,
    source_commit: str | None = None,
    max_workers: int | None = None,
    invocation: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Exhaustively compare compiled and direct-reference shared-consumer pairs."""

    contract = load_operator_template_coupling_contract()
    source_gate = _load_source_gate(contract)
    destination = Path(output_path).resolve()
    source_before = _source_manifest(source_commit, extra_snapshot_paths=(_RUNNER,))
    specs = _assignment_specs(contract)
    required_workers = contract["max_workers"]
    if max_workers is not None and max_workers != required_workers:
        raise ValueError(f"Gate 1A-v4 requires exactly {required_workers} parallel workers")
    worker_count = min(required_workers, len(specs))
    accelerator = load_gate1a_accelerator(contract["hardware"])
    started = perf_counter()
    with TemporaryDirectory(prefix="stream-gate1a-v4-") as temporary:
        root = Path(temporary)
        results = _run_assignments(specs, root / "parallel", accelerator, contract, worker_count)
        serial_results = _run_assignments(specs, root / "serial", accelerator, contract, 1)
    replay_exact = results == serial_results
    source_after = _source_manifest(source_commit, extra_snapshot_paths=(_RUNNER,))
    source = _source_run_manifest(source_before, source_after, destination)
    static_review = _static_review_manifest(contract, source)
    observations = _observations(
        contract,
        source_gate,
        source,
        static_review,
        specs,
        results,
        replay_exact,
        worker_count,
    )
    expected = contract["pass_criteria"]
    if set(observations) != set(expected):
        raise OperatorTemplateCouplingError("contract criteria do not match computed observations")
    criteria = {name: observations[name] == value for name, value in expected.items()}
    payload = {
        "contract": contract,
        "verdict": "PASS" if all(criteria.values()) else "FAIL",
        "classification": "COUPLED_LEGALITY",
        "evidence_class": "MICRO_EXACT_BINARY_COMPATIBILITY",
        "criteria": criteria,
        "observations": observations,
        "source_gate": source_gate,
        "source": source,
        "environment": _environment_manifest(),
        "static_review": static_review,
        "execution": {
            "assignment_count": len(specs),
            "invocation": list(invocation) if invocation is not None else None,
            "worker_count": worker_count,
            "serial_replay_worker_count": 1,
            "wall_seconds": round(perf_counter() - started, 6),
            "tta_solve_invoked": False,
            "serial_parallel_replay_exact": replay_exact,
        },
        "status_semantics": {
            "EXACT_EXECUTABLE": "exactly prepared through timeslot construction; no TTA feasibility solve",
            "UNSUPPORTED": "both compiler and direct reference reject the registered structural combination",
        },
        "factor_graph": {
            "variable_count": 4,
            "non_unary_factor_count": 2,
            "factor_arity_histogram": {"2": 2},
            "largest_coupled_component": 2,
            "induced_width": 1,
            "factor_semantics": "shared-tensor tiling compatibility only",
        },
        "summary": _summary(results),
        "assignments": results,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(destination)
    return payload


def write_operator_template_coupling_provenance(
    report_path: str | Path,
    report: dict[str, Any],
    invocation: tuple[str, ...],
    stdout_text: str,
    stderr_text: str,
) -> Path:
    """Write post-run report/log hashes beside a completed Gate artifact."""

    report_path = Path(report_path).resolve()
    stdout_path = report_path.with_suffix(report_path.suffix + ".stdout.log")
    stderr_path = report_path.with_suffix(report_path.suffix + ".stderr.log")
    manifest_path = report_path.with_suffix(report_path.suffix + ".run.json")
    _atomic_write_text(stdout_path, stdout_text)
    _atomic_write_text(stderr_path, stderr_text)
    environment = report["environment"]
    manifest = {
        "schema": "stream-gate-run-provenance-v1",
        "host": platform.node(),
        "source_commit": report["source"]["commit"],
        "invocation": list(invocation),
        "report": {"path": str(report_path), "sha256": _plain_file_digest(report_path)},
        "stdout": {"path": str(stdout_path), "sha256": _plain_file_digest(stdout_path)},
        "stderr": {"path": str(stderr_path), "sha256": _plain_file_digest(stderr_path)},
        "environment": environment,
        "environment_sha256": sha256(
            json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def verify_operator_template_coupling_provenance(manifest_path: str | Path) -> bool:
    """Accept a formal run only when its completion marker and every recorded hash match."""

    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest["schema"] != "stream-gate-run-provenance-v1" or not manifest["source_commit"]:
            return False
        for artifact in ("report", "stdout", "stderr"):
            entry = manifest[artifact]
            if _plain_file_digest(Path(entry["path"])) != entry["sha256"]:
                return False
        environment_digest = sha256(
            json.dumps(manifest["environment"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return environment_digest == manifest["environment_sha256"]
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _plain_file_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _assignment_specs(contract: dict[str, Any]) -> list[tuple[str, str, str, int, int]]:
    template_count = len(contract["template_family"])
    specs = [
        (dag_class, targets[0], targets[1], left, right)
        for dag_class, targets in contract["coupled_pairs"].items()
        for left in range(template_count)
        for right in range(template_count)
    ]
    if len(specs) != contract["expected_assignment_count"]:
        raise OperatorTemplateCouplingError("assignment denominator does not match the frozen contract")
    return specs


def _run_assignments(specs, temporary_root, accelerator, contract, worker_count) -> list[dict[str, Any]]:
    results: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="gate1a-v4") as executor:
        pending = {}
        for spec in specs:
            dag_class, left_target, right_target, left_index, right_index = spec
            targets = (left_target, right_target)
            template_indices = (left_index, right_index)
            future = executor.submit(
                _run_assignment,
                dag_class,
                targets,
                template_indices,
                temporary_root / f"{dag_class}-{template_indices[0]}-{template_indices[1]}",
                accelerator,
                contract,
            )
            pending[future] = spec
        for future in as_completed(pending):
            results[pending[future]] = future.result()
    return [results[spec] for spec in specs]


def _run_assignment(dag_class, targets, template_indices, output_root, accelerator, contract) -> dict[str, Any]:
    workload = build_gate1a_workload(dag_class)
    baseline = _baseline_mapping(workload, accelerator)
    library = _template_library(workload, contract)
    templates = tuple(
        _template_from_spec(target, contract["template_family"][index])
        for target, index in zip(targets, template_indices, strict=True)
    )
    assignment_id = _assignment_id(dag_class, targets, template_indices)
    assignment = OperatorTemplateAssignment(assignment_id, templates)
    compiled_context = _run_tiling_pipeline(
        workload,
        baseline,
        output_root / "compiled",
        assignment=assignment,
        library=library,
    )
    reference_mapping = _direct_reference_mapping_for_templates(workload, baseline, templates)
    reference_context = _run_tiling_pipeline(workload, reference_mapping, output_root / "reference")
    compilation = compiled_context.get("operator_template_compilation")
    if compilation is None:
        raise OperatorTemplateCouplingError(f"{assignment_id}: compiler stage did not record a result")

    compiled_pairs = {target: _selected_pair(compiled_context.get("mapping"), target) for target in targets}
    reference_pairs = {target: _selected_pair(reference_context.get("mapping"), target) for target in targets}
    expected_pairs = {template.target: _expected_pair(template) for template in templates}
    compiled_outcome = _prepared_problem_outcome(compiled_context, accelerator, output_root / "compiled-scheduler")
    reference_outcome = _prepared_problem_outcome(reference_context, accelerator, output_root / "reference-scheduler")
    hashes = [
        _run_compiler_only(workload, baseline, assignment, library).semantic_hash
        for _ in range(contract["compile_repetitions"])
    ]
    pair_exact = compiled_pairs == reference_pairs == expected_pairs
    outcome_exact = compiled_outcome == reference_outcome
    exact_executable = compiled_outcome["status"] == "EXACT_EXECUTABLE"
    predicted_compatible = _templates_are_compatible(templates)
    baseline_templates = tuple(
        _template_from_spec(target, contract["template_family"][-1]) for target in targets
    )
    baseline_round_trip = None
    if templates == baseline_templates:
        baseline_round_trip = (
            canonical_mapping_manifest(reference_mapping) == canonical_mapping_manifest(baseline)
            and outcome_exact
            and exact_executable
        )
    return {
        "assignment_id": assignment_id,
        "dag_class": dag_class,
        "targets": list(targets),
        "template_indices": list(template_indices),
        "templates": [template.key for template in templates],
        "paired_candidate_set_exact": pair_exact,
        "compiled_pairs": compiled_pairs,
        "reference_pairs": reference_pairs,
        "expected_pairs": expected_pairs,
        "compiled_reference_outcome_exact": outcome_exact,
        "compiled_outcome": compiled_outcome,
        "reference_outcome": reference_outcome,
        "predicted_compatible": predicted_compatible,
        "compatibility_prediction_exact": predicted_compatible == exact_executable,
        "exact_executable": exact_executable,
        "joint_recovery": all(index in (4, 5) for index in template_indices) and exact_executable,
        "compiler_hashes": hashes,
        "compiler_deterministic": len(set(hashes)) == 1,
        "baseline_round_trip_exact": baseline_round_trip,
        "silent_relaxations": int(not pair_exact) + int(not outcome_exact),
    }


def _templates_are_compatible(templates) -> bool:
    partitioned = {template.splits for template in templates if template.splits}
    return len(partitioned) <= 1


def _assignment_id(dag_class, targets, template_indices) -> str:
    selections = ",".join(
        f"{target}={index}" for target, index in zip(targets, template_indices, strict=True)
    )
    return f"{dag_class}:{selections}"


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    unsupported_reasons: dict[str, int] = {}
    per_dag: dict[str, dict[str, int]] = {}
    for result in results:
        outcome = result["compiled_outcome"]
        if outcome["status"] == "UNSUPPORTED":
            reason = outcome["reason"]
            unsupported_reasons[reason] = unsupported_reasons.get(reason, 0) + 1
        dag = per_dag.setdefault(result["dag_class"], {"assignments": 0, "executable": 0, "unsupported": 0})
        dag["assignments"] += 1
        dag["executable"] += int(result["exact_executable"])
        dag["unsupported"] += int(not result["exact_executable"])
    return {
        "assignment_count": len(results),
        "exact_executable_count": sum(result["exact_executable"] for result in results),
        "unsupported_count": sum(not result["exact_executable"] for result in results),
        "unsupported_reason_counts": dict(sorted(unsupported_reasons.items())),
        "paired_candidate_set_exact_count": sum(result["paired_candidate_set_exact"] for result in results),
        "compiled_reference_outcome_exact_count": sum(
            result["compiled_reference_outcome_exact"] for result in results
        ),
        "compatibility_prediction_exact_count": sum(
            result["compatibility_prediction_exact"] for result in results
        ),
        "joint_recovery_count": sum(result["joint_recovery"] for result in results),
        "compiler_deterministic_count": sum(result["compiler_deterministic"] for result in results),
        "baseline_round_trip_check_count": sum(
            result["baseline_round_trip_exact"] is not None for result in results
        ),
        "baseline_round_trip_exact_count": sum(
            result["baseline_round_trip_exact"] is True for result in results
        ),
        "preparation_failure_count": sum(
            result["compiled_outcome"]["status"] == "PREPARATION_FAILURE" for result in results
        ),
        "silent_relaxation_count": sum(result["silent_relaxations"] for result in results),
        "per_dag": per_dag,
    }


def _relation_not_cartesian(results: list[dict[str, Any]], coupled_pairs) -> bool:
    by_id = {result["assignment_id"]: result for result in results}
    for dag_class, targets in coupled_pairs.items():
        values = {
            pair: by_id[_assignment_id(dag_class, targets, pair)]["exact_executable"]
            for pair in ((4, 4), (6, 6), (4, 6), (6, 4))
        }
        if values != {(4, 4): True, (6, 6): True, (4, 6): False, (6, 4): False}:
            return False
    return True


def _observations(contract, source_gate, source, static_review, specs, results, replay_exact, worker_count):
    summary = _summary(results)
    unsupported = sorted(result["assignment_id"] for result in results if not result["exact_executable"])
    return {
        "source_gate_hash_matches": source_gate["hash_matches"],
        "source_identified": source["identified"],
        "static_review_scope_match": static_review["scope_match"],
        "assignment_denominator_complete": len(results) == len(specs) == contract["expected_assignment_count"],
        "paired_candidate_set_exact": summary["paired_candidate_set_exact_count"] == len(specs),
        "compiled_reference_outcome_exact": summary["compiled_reference_outcome_exact_count"] == len(specs),
        "compatibility_prediction_exact": summary["compatibility_prediction_exact_count"] == len(specs),
        "parallel_worker_count_exact": worker_count == contract["max_workers"],
        "downstream_serial_parallel_replay_exact": replay_exact,
        "compiler_deterministic": summary["compiler_deterministic_count"] == len(specs),
        "baseline_round_trip_exact": summary["baseline_round_trip_check_count"]
        == summary["baseline_round_trip_exact_count"]
        == contract["expected_baseline_round_trip_count"],
        "unsupported_assignment_set_exact": unsupported == contract["expected_unsupported_assignments"],
        "unsupported_reason_counts_exact": summary["unsupported_reason_counts"]
        == contract["expected_unsupported_reason_counts"],
        "exact_executable_count_matches": summary["exact_executable_count"]
        == contract["expected_exact_executable_count"],
        "joint_recovery_count_matches": summary["joint_recovery_count"] == contract["expected_joint_recovery_count"],
        "binary_relation_not_cartesian": _relation_not_cartesian(results, contract["coupled_pairs"]),
        "unexpected_preparation_failure_count": summary["preparation_failure_count"],
        "silent_relaxation_count": summary["silent_relaxation_count"],
    }


def _load_source_gate(contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["source_gate"]
    path = Path(expected["artifact"])
    digest = _file_digest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    hash_matches = (
        digest == expected["sha256"]
        and payload.get("verdict") == expected["required_verdict"]
        and payload.get("evidence_class") == expected["required_evidence_class"]
    )
    if not hash_matches:
        raise OperatorTemplateCouplingError("Gate 1A-v3 source evidence does not match the frozen contract")
    singleton_unsupported = payload["summary"]["unsupported_reason_counts"].get(
        "SHARED_INPUT_TILING_INCOMPATIBLE", 0
    )
    if singleton_unsupported != expected["required_singleton_unsupported_count"]:
        raise OperatorTemplateCouplingError("source Gate does not contain the frozen singleton incompatibilities")
    return {
        "path": str(path),
        "sha256": digest,
        "verdict": payload["verdict"],
        "evidence_class": payload["evidence_class"],
        "singleton_unsupported_count": singleton_unsupported,
        "hash_matches": True,
    }


def _validate_contract(contract: dict[str, Any]) -> None:
    expected_keys = {
        "contract",
        "version",
        "source_gate",
        "coupled_pairs",
        "template_family",
        "assignment_policy",
        "artifact_acceptance",
        "compatibility_rule",
        "reference_pipeline",
        "compile_repetitions",
        "expected_assignment_count",
        "expected_baseline_round_trip_count",
        "expected_exact_executable_count",
        "expected_joint_recovery_count",
        "expected_unsupported_assignments",
        "expected_unsupported_reason_counts",
        "max_workers",
        "hardware",
        "static_review",
        "pass_criteria",
        "excluded_claims",
    }
    if set(contract) != expected_keys or contract["version"] != _CONTRACT_VERSION:
        raise OperatorTemplateCouplingError("invalid Gate 1A-v4 contract schema")
    templates = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in contract["template_family"]]
    if not templates or len(templates) != len(set(templates)):
        raise OperatorTemplateCouplingError("template family must be non-empty and unique")
    if any(len(targets) != 2 or len(set(targets)) != 2 for targets in contract["coupled_pairs"].values()):
        raise OperatorTemplateCouplingError("each coupled factor must name exactly two distinct targets")
    assignment_count = len(contract["coupled_pairs"]) * len(templates) ** 2
    if assignment_count != contract["expected_assignment_count"]:
        raise OperatorTemplateCouplingError("assignment count is inconsistent with the frozen cross product")
    unsupported = contract["expected_unsupported_assignments"]
    if unsupported != sorted(set(unsupported)):
        raise OperatorTemplateCouplingError("unsupported assignment IDs must be sorted and unique")
    if sum(contract["expected_unsupported_reason_counts"].values()) != len(unsupported):
        raise OperatorTemplateCouplingError("unsupported reason counts do not cover the frozen assignments")
    if contract["expected_exact_executable_count"] + len(unsupported) != assignment_count:
        raise OperatorTemplateCouplingError("executable and unsupported counts do not cover the denominator")
    if contract["expected_baseline_round_trip_count"] != len(contract["coupled_pairs"]):
        raise OperatorTemplateCouplingError("baseline checks must cover every coupled factor")
    if contract["compile_repetitions"] < 2 or contract["max_workers"] <= 1:
        raise OperatorTemplateCouplingError("repetition and worker counts must be nontrivial")
    expected_criteria = {
        "source_gate_hash_matches",
        "source_identified",
        "static_review_scope_match",
        "assignment_denominator_complete",
        "paired_candidate_set_exact",
        "compiled_reference_outcome_exact",
        "compatibility_prediction_exact",
        "parallel_worker_count_exact",
        "downstream_serial_parallel_replay_exact",
        "compiler_deterministic",
        "baseline_round_trip_exact",
        "unsupported_assignment_set_exact",
        "unsupported_reason_counts_exact",
        "exact_executable_count_matches",
        "joint_recovery_count_matches",
        "binary_relation_not_cartesian",
        "unexpected_preparation_failure_count",
        "silent_relaxation_count",
    }
    if set(contract["pass_criteria"]) != expected_criteria:
        raise OperatorTemplateCouplingError("pass criteria are missing, unknown, or unevaluated")


__all__ = [
    "OperatorTemplateCouplingError",
    "load_operator_template_coupling_contract",
    "run_operator_template_coupling",
    "verify_operator_template_coupling_provenance",
    "write_operator_template_coupling_provenance",
]
