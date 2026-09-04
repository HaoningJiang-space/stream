"""Complete TETRA oracle census for Gate 1B structural predictive value."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from typing import Any

from stream.cost_model.communication_manager import MulticastPathPlan
from stream.opt.allocation.constraint_optimization.tensor_restriction import (
    TensorRestriction,
    TransferPlanRestriction,
    restriction_manifest,
    tensor_placement_key,
    transfer_plan_key,
)
from stream.opt.solver import SolverParams
from stream.structural.gate1b_cases import build_gate1b_scheduler, load_gate1b_contract
from stream.structural.pipeline import (
    prepare_tensor_restricted_problem,
    prepare_uninstrumented_reference_problem,
)
from stream.structural.stream_contract import Gate1AEvalConfig
from stream.workload.node import TransferNode
from stream.workload.tensor import Tensor

_MIN_ADJACENT_TRANSFERS = 2
_MIN_CORRELATION_SAMPLES = 2
_SHA256_DIGEST_LENGTH = len("sha256:") + 64


@dataclass(frozen=True, slots=True)
class StructuralCandidate:
    candidate_id: str
    restrictions: tuple[TensorRestriction, ...]
    structural_score: int
    score_events: tuple[dict[str, Any], ...]


def run_predictive_gate(
    output_path: str | Path,
    *,
    container_image: str | Path,
    oci_source_digest: str,
) -> dict[str, Any]:
    """Run the preregistered full census and atomically write its evidence artifact."""

    contract = load_gate1b_contract()
    source_gate = _source_gate_manifest(contract)
    environment = _environment_manifest(container_image, oci_source_digest, contract)
    eval_config = _eval_config(contract)
    class_results = {}
    for dag_class in contract["dag_classes"]:
        print(f"Gate 1B: preparing {dag_class}", flush=True)
        candidates = enumerate_structural_candidates(dag_class, eval_config, contract)
        baseline = _evaluate_baseline(dag_class, eval_config)
        evaluations = []
        for index, candidate in enumerate(candidates, start=1):
            evaluations.append(_evaluate_candidate(dag_class, candidate, eval_config))
            if index % 16 == 0:
                print(f"Gate 1B: {dag_class} {index}/{len(candidates)}", flush=True)
        class_results[dag_class] = _class_result(dag_class, candidates, baseline, evaluations, contract)

    criteria = contract["pass_criteria"]
    passing_classes = sum(result["pass"] for result in class_results.values())
    validity_ok = all(result["validity_ok"] for result in class_results.values())
    passed = validity_ok and passing_classes >= int(criteria["minimum_passing_dag_classes"])
    payload = {
        "contract": contract["contract"],
        "version": contract["version"],
        "verdict": "PASS" if passed else "FAIL",
        "source_gate": source_gate,
        "structural_objective": contract["structural_objective"],
        "metric_contract": contract["metrics"],
        "pass_criteria": criteria,
        "passing_dag_classes": passing_classes,
        "required_passing_dag_classes": criteria["minimum_passing_dag_classes"],
        "evaluation_coverage": sum(result["evaluated_candidates"] for result in class_results.values())
        / sum(result["candidate_count"] for result in class_results.values()),
        "classes": class_results,
        "environment": environment,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


def enumerate_structural_candidates(
    dag_class: str,
    eval_config: Gate1AEvalConfig,
    contract: dict[str, Any] | None = None,
) -> tuple[StructuralCandidate, ...]:
    """Enumerate every concrete singleton placement/path assignment in the frozen space."""

    contract = contract or load_gate1b_contract()
    reference = prepare_uninstrumented_reference_problem(build_gate1b_scheduler(dag_class), eval_config)
    tta = reference.build_tta()
    targets = _staging_targets(tta)
    required_targets = int(contract["candidate_space"]["required_staging_tensors_per_class"])
    if len(targets) != required_targets:
        raise RuntimeError(
            f"{dag_class}: expected {required_targets} restrictable staging tensors, found {len(targets)}"
        )

    choices_by_tensor = tuple(_tensor_choices(tta, tensor, placements) for tensor, placements in targets)
    candidates = []
    for choices in product(*choices_by_tensor):
        restrictions = tuple(choice[0] for choice in choices)
        events = tuple(event for choice in choices for event in choice[1])
        event_ids = [event["event_id"] for event in events]
        if len(event_ids) != len(set(event_ids)):
            raise RuntimeError(f"{dag_class}: a physical link-service event has multiple owners")
        manifest = restriction_manifest(restrictions)
        candidate_id = _digest({"dag_class": dag_class, "restrictions": manifest})
        candidates.append(
            StructuralCandidate(candidate_id, restrictions, sum(event["cycles"] for event in events), events)
        )
    candidates.sort(key=lambda candidate: candidate.candidate_id)
    expected = int(contract["candidate_space"]["required_candidates_per_class"])
    if len(candidates) != expected or len({candidate.candidate_id for candidate in candidates}) != expected:
        raise RuntimeError(f"{dag_class}: expected {expected} unique candidates, found {len(candidates)}")
    return tuple(candidates)


def link_service_events(transfer: TransferNode, path: MulticastPathPlan) -> tuple[dict[str, Any], ...]:
    """Return the independently auditable link-time work charged to one transfer plan."""

    if len(transfer.inputs) != 1:
        raise ValueError(f"{transfer.name}: structural score supports one transfer input")
    tensor_bits = int(transfer.inputs[0].size_bits())
    chains = len(path.targets) if 1 < len(path.sources) == len(path.targets) else 1
    events = []
    link_ids = set()
    for link_index, link in enumerate(path.links_used):
        if link.bandwidth <= 0:
            raise ValueError(f"{transfer.name}: link bandwidth must be positive")
        link_id = (
            _resource_id(link.sender),
            _resource_id(link.receiver),
            link.bandwidth,
            link.bidirectional,
        )
        if link_id in link_ids:
            raise ValueError(f"{transfer.name}: path repeats a physical link")
        link_ids.add(link_id)
        events.append(
            {
                "event_id": f"{transfer.name}:link:{link_index}:{_digest(link_id)[:12]}",
                "transfer": transfer.name,
                "tensor_bits": tensor_bits,
                "parallel_chains": chains,
                "sender": _resource_id(link.sender),
                "receiver": _resource_id(link.receiver),
                "bandwidth_bits_per_cycle": link.bandwidth,
                "cycles": math.ceil(tensor_bits / (link.bandwidth * chains)),
            }
        )
    return tuple(events)


def predictive_metrics(
    evaluations: tuple[dict[str, Any], ...],
    top_ns: tuple[int, ...],
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    """Compute deterministic and latency-tie-aware retention metrics."""

    if not evaluations or any(item.get("latency_cycles") is None for item in evaluations):
        raise ValueError("predictive metrics require a complete latency census")
    structural_order = sorted(evaluations, key=lambda item: (item["structural_score"], item["candidate_id"]))
    tetra_order = sorted(evaluations, key=lambda item: (item["latency_cycles"], item["candidate_id"]))
    optimum = int(tetra_order[0]["latency_cycles"])
    recall = {}
    regret = {}
    selections = {}
    for n in top_ns:
        selected = structural_order[:n]
        selected_ids = {item["candidate_id"] for item in selected}
        selections[str(n)] = [item["candidate_id"] for item in selected]
        best = min(int(item["latency_cycles"]) for item in selected)
        regret[str(n)] = (best - optimum) / optimum if optimum else (0.0 if best == 0 else math.inf)
        recall[str(n)] = {}
        for k in top_ks:
            cutoff = int(tetra_order[k - 1]["latency_cycles"])
            good_ids = {item["candidate_id"] for item in tetra_order if int(item["latency_cycles"]) <= cutoff}
            retained = len(selected_ids & good_ids)
            recall[str(n)][str(k)] = {
                "value": min(1.0, retained / k),
                "retained": retained,
                "latency_cutoff_cycles": cutoff,
                "tie_expanded_good_set_size": len(good_ids),
            }
    return {
        "optimum_latency_cycles": optimum,
        "recall_at_n": recall,
        "regret_at_n": regret,
        "spearman": _spearman(
            tuple(float(item["structural_score"]) for item in evaluations),
            tuple(float(item["latency_cycles"]) for item in evaluations),
        ),
        "structural_top_n": selections,
        "tetra_order": [item["candidate_id"] for item in tetra_order],
    }


def _staging_targets(tta) -> tuple[tuple[Tensor, tuple], ...]:
    targets = []
    for tensor, placements in tta.possible_tensor_allocations.items():
        producers = tuple(transfer for transfer in tta.transfer_nodes if tensor in transfer.outputs)
        adjacent = tuple(transfer for transfer in tta.transfer_nodes if tensor in transfer.tensors)
        if len(placements) > 1 and len(producers) == 1 and len(adjacent) >= _MIN_ADJACENT_TRANSFERS:
            targets.append((tensor, tuple(sorted(placements, key=tensor_placement_key))))
    return tuple(sorted(targets, key=lambda item: item[0].name))


def _tensor_choices(tta, tensor: Tensor, placements: tuple) -> tuple[tuple[TensorRestriction, tuple], ...]:
    adjacent = tuple(
        sorted((transfer for transfer in tta.transfer_nodes if tensor in transfer.tensors), key=lambda x: x.name)
    )
    choices = []
    for placement in placements:
        placement_key = tensor_placement_key(placement)
        paths_by_transfer = []
        for transfer in adjacent:
            compatible = tuple(
                sorted(
                    (
                        path
                        for path in tta.possible_transfer_allocations[transfer]
                        if _path_endpoint_placement_key(path, transfer, tensor) == placement_key
                    ),
                    key=transfer_plan_key,
                )
            )
            if not compatible:
                raise RuntimeError(f"{transfer.name}: no path realizes placement {placement_key}")
            paths_by_transfer.append((transfer, compatible))
        for path_choices in product(*(paths for _, paths in paths_by_transfer)):
            transfer_restrictions = tuple(
                TransferPlanRestriction(transfer.name, frozenset({transfer_plan_key(path)}))
                for (transfer, _), path in zip(paths_by_transfer, path_choices, strict=True)
            )
            restriction = TensorRestriction(tensor.name, frozenset({placement_key}), transfer_restrictions)
            events = tuple(
                event
                for (transfer, _), path in zip(paths_by_transfer, path_choices, strict=True)
                for event in link_service_events(transfer, path)
            )
            choices.append((restriction, events))
    return tuple(choices)


def _path_endpoint_placement_key(path: MulticastPathPlan, transfer: TransferNode, tensor: Tensor) -> str:
    if tensor in transfer.inputs:
        return tensor_placement_key(tuple(path.sources))
    if tensor in transfer.outputs:
        return tensor_placement_key(tuple(path.targets))
    raise RuntimeError(f"{transfer.name} is not adjacent to {tensor.name}")


def _evaluate_baseline(dag_class: str, eval_config: Gate1AEvalConfig) -> dict[str, Any]:
    prepared = prepare_uninstrumented_reference_problem(build_gate1b_scheduler(dag_class), eval_config)
    return _solve(prepared, candidate_id="baseline", structural_score=None, require_exact_domains=False)


def _evaluate_candidate(
    dag_class: str,
    candidate: StructuralCandidate,
    eval_config: Gate1AEvalConfig,
) -> dict[str, Any]:
    try:
        prepared = prepare_tensor_restricted_problem(
            build_gate1b_scheduler(dag_class),
            candidate.restrictions,
            eval_config,
        )
        result = _solve(
            prepared,
            candidate_id=candidate.candidate_id,
            structural_score=candidate.structural_score,
            require_exact_domains=True,
        )
        result["restriction_manifest"] = restriction_manifest(candidate.restrictions)
        result["score_events"] = candidate.score_events
        return result
    except Exception as error:  # noqa: BLE001 - Gate artifact must preserve every failed candidate
        return {
            "candidate_id": candidate.candidate_id,
            "structural_score": candidate.structural_score,
            "status": "ERROR",
            "latency_cycles": None,
            "domain_exact": False,
            "evaluation_invariant_hash": None,
            "error": f"{type(error).__name__}: {error}",
            "restriction_manifest": restriction_manifest(candidate.restrictions),
            "score_events": candidate.score_events,
        }


def _solve(prepared, *, candidate_id: str, structural_score: int | None, require_exact_domains: bool) -> dict[str, Any]:
    tta = prepared.build_tta()
    domain_exact = _restriction_domains_exact(tta, prepared.restrictions) if require_exact_domains else True
    tta.model.set_param(SolverParams.VERBOSITY, 0)
    tta.model.optimize()
    stats = tta.model.solve_stats()
    latency = None
    if stats.status == "OPTIMAL":
        raw_latency = float(tta.total_latency.X)
        if not raw_latency.is_integer():
            raise RuntimeError(f"TTA returned non-integral total latency {raw_latency}")
        latency = int(raw_latency)
    pipeline_manifest = (
        prepared.pipeline_manifest if hasattr(prepared, "pipeline_manifest") else prepared.compilation.pipeline_manifest
    )
    return {
        "candidate_id": candidate_id,
        "structural_score": structural_score,
        "status": stats.status,
        "latency_cycles": latency,
        "solver_objective": stats.objective,
        "solve_time_s": stats.solve_time_s,
        "mip_gap": stats.mip_gap,
        "domain_exact": domain_exact,
        "problem_hash": (
            prepared.problem_hash if hasattr(prepared, "problem_hash") else prepared.compilation.problem_hash
        ),
        "evaluation_invariant_hash": _evaluation_invariant_hash(pipeline_manifest),
        "timeslot_hash": _digest(asdict(pipeline_manifest)["timeslots"]),
    }


def _restriction_domains_exact(tta, restrictions: tuple[TensorRestriction, ...]) -> bool:
    tensors = {tensor.name: tensor for tensor in tta.possible_tensor_allocations}
    transfers = {transfer.name: transfer for transfer in tta.possible_transfer_allocations}
    for restriction in restrictions:
        tensor = tensors.get(restriction.tensor_id)
        if tensor is None or restriction.allowed_placements is None:
            return False
        observed_placements = frozenset(tensor_placement_key(x) for x in tta.possible_tensor_allocations[tensor])
        if observed_placements != restriction.allowed_placements or len(observed_placements) != 1:
            return False
        for transfer_restriction in restriction.allowed_transfer_plans:
            transfer = transfers.get(transfer_restriction.transfer_id)
            if transfer is None:
                return False
            observed_paths = frozenset(transfer_plan_key(x) for x in tta.possible_transfer_allocations[transfer])
            if observed_paths != transfer_restriction.allowed_plans or len(observed_paths) != 1:
                return False
    return True


def _class_result(
    dag_class: str,
    candidates: tuple[StructuralCandidate, ...],
    baseline: dict[str, Any],
    evaluations: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    accepted = contract["evaluation"]["accepted_status"]
    expected_count = int(contract["candidate_space"]["required_candidates_per_class"])
    valid_evaluations = tuple(item for item in evaluations if item["status"] == accepted)
    invariant_hashes = {item["evaluation_invariant_hash"] for item in evaluations}
    invariant_hashes.add(baseline["evaluation_invariant_hash"])
    validity_ok = (
        len(candidates) == expected_count
        and len(evaluations) == expected_count
        and len(valid_evaluations) == expected_count
        and all(item["domain_exact"] for item in evaluations)
        and baseline["status"] == accepted
        and len(invariant_hashes) == 1
        and expected_count > max(contract["metrics"]["top_n"])
    )
    metrics = None
    class_pass = False
    if validity_ok:
        metrics = predictive_metrics(
            valid_evaluations,
            tuple(int(value) for value in contract["metrics"]["top_n"]),
            tuple(int(value) for value in contract["metrics"]["tetra_top_k"]),
        )
        recall = metrics["recall_at_n"]["16"]["5"]["value"]
        regret = metrics["regret_at_n"]["16"]
        class_pass = recall >= float(contract["pass_criteria"]["recall_at_16_tetra_top_5_min"]) and regret <= float(
            contract["pass_criteria"]["regret_at_16_max"]
        )
        baseline_latency = int(baseline["latency_cycles"])
        top16_ids = set(metrics["structural_top_n"]["16"])
        best_top16_latency = min(
            item["latency_cycles"] for item in valid_evaluations if item["candidate_id"] in top16_ids
        )
        metrics["best_top16_plus_baseline_latency_cycles"] = min(baseline_latency, best_top16_latency)
        metrics["baseline_non_regression"] = metrics["best_top16_plus_baseline_latency_cycles"] <= baseline_latency
    return {
        "dag_class": dag_class,
        "candidate_count": len(candidates),
        "evaluated_candidates": len(valid_evaluations),
        "evaluation_coverage": len(valid_evaluations) / len(candidates),
        "validity_ok": validity_ok,
        "pass": class_pass,
        "baseline": baseline,
        "metrics": metrics,
        "evaluations": evaluations,
    }


def _eval_config(contract: dict[str, Any]) -> Gate1AEvalConfig:
    evaluation = contract["evaluation"]
    return Gate1AEvalConfig(
        backend=evaluation["backend"],
        constraints=tuple(evaluation["constraints"]),
        timeslot_policy=evaluation["timeslot_policy"],
        time_limit_s=evaluation["time_limit_s"],
        solver_params=tuple(sorted(evaluation["solver_params"].items())),
    )


def _source_gate_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source_contract"]
    path = Path(source["artifact"])
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("contract") != source["name"]
        or report.get("verdict") != source["required_verdict"]
        or report.get("version") != source["version"]
    ):
        raise RuntimeError("Gate 1A-v2 source contract is not the frozen PASS artifact")
    return {
        "name": source["name"],
        "version": source["version"],
        "verdict": report["verdict"],
        "artifact": str(path),
        "artifact_sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _environment_manifest(
    container_image: str | Path,
    oci_source_digest: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    ortools_version = version("ortools")
    if _version_tuple(ortools_version) < _version_tuple(contract["evaluation"]["minimum_ortools_version"]):
        raise RuntimeError(f"Gate 1B requires OR-Tools >= {contract['evaluation']['minimum_ortools_version']}")
    image = Path(container_image)
    if not image.is_file():
        raise FileNotFoundError(f"container image not found: {image}")
    if not oci_source_digest.startswith("sha256:") or len(oci_source_digest) != _SHA256_DIGEST_LENGTH:
        raise ValueError("OCI source digest must be a sha256 digest")
    source_paths = (
        "stream/cost_model/communication_manager.py",
        "stream/cost_model/steady_state_scheduler.py",
        "stream/opt/allocation/constraint_optimization/tensor_restriction.py",
        "stream/opt/allocation/constraint_optimization/transfer_and_tensor_allocation.py",
        "stream/opt/allocation/constraint_optimization/timeslot_allocation.py",
        "stream/opt/allocation/constraint_optimization/utils.py",
        "stream/opt/solver/solver.py",
        "stream/structural/gate1b_cases.py",
        "stream/structural/pipeline.py",
        "stream/structural/predictive_gate.py",
        "stream/structural/stream_contract.py",
        "stream/structural/contracts/gate1b_contract.json",
        "stream/inputs/examples/hardware/tpu_v7_ironwood.yaml",
        "stream/inputs/examples/hardware/cores/tpu_v7_hbm.yaml",
        "stream/inputs/examples/hardware/cores/tpu_v7_mxu.yaml",
        "stream/inputs/examples/hardware/cores/tpu_v7_vmem.yaml",
        "stream/inputs/examples/hardware/cores/tpu_v7_vpu.yaml",
        "docs/source/structural_predictive_gate1b.md",
        "pyproject.toml",
    )
    packages = {}
    for package in ("ortools", "stream-dse", "xdsl", "zigzag-dse"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "completion_marker": "COMPLETE",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libc": platform.libc_ver(),
        "packages": packages,
        "declared_dependencies": pyproject["project"]["dependencies"],
        "pip_freeze_sha256": sha256(freeze.encode()).hexdigest(),
        "container": {
            "filename": image.name,
            "sif_sha256": sha256(image.read_bytes()).hexdigest(),
            "oci_source_digest": oci_source_digest,
        },
        "source_sha256": {path: sha256(Path(path).read_bytes()).hexdigest() for path in source_paths},
    }


def _evaluation_invariant_hash(manifest) -> str:
    value = asdict(manifest)
    value.pop("timeslots")
    value.pop("candidate_correlations")
    return _digest(value)


def _spearman(xs: tuple[float, ...], ys: tuple[float, ...]) -> float | None:
    if len(xs) != len(ys) or len(xs) < _MIN_CORRELATION_SAMPLES:
        raise ValueError("Spearman inputs must have equal length of at least two")
    x_ranks, y_ranks = _average_ranks(xs), _average_ranks(ys)
    x_mean = sum(x_ranks) / len(x_ranks)
    y_mean = sum(y_ranks) / len(y_ranks)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_ranks, y_ranks, strict=True))
    x_norm = math.sqrt(sum((x - x_mean) ** 2 for x in x_ranks))
    y_norm = math.sqrt(sum((y - y_mean) ** 2 for y in y_ranks))
    return None if x_norm == 0 or y_norm == 0 else numerator / (x_norm * y_norm)


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average = (start + 1 + end) / 2
        for index, _ in indexed[start:end]:
            ranks[index] = average
        start = end
    return tuple(ranks)


def _version_tuple(value: str) -> tuple[int, ...]:
    pieces = []
    for piece in value.split("."):
        digits = "".join(character for character in piece if character.isdigit())
        if not digits:
            break
        pieces.append(int(digits))
    return tuple(pieces)


def _resource_id(resource) -> str:
    if isinstance(resource, str):
        return f"str:{resource}"
    if hasattr(resource, "id"):
        return f"{type(resource).__name__}:{resource.id}"
    raise TypeError(f"unsupported path endpoint type: {type(resource).__name__}")


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
