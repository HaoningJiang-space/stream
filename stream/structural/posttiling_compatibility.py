"""Gate 2F-B: production faithfulness of pre-tiling shared-tensor compatibility."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from stream.cost_model.steady_state_scheduler import (
    SharedInputTilingIncompatibilityError,
    SteadyStateScheduler,
    TensorRelevantTilingDecision,
    TransferDomainIncompatibilityError,
    TransferLineage,
)
from stream.execution_boundary import ExecutionEvent, ForbiddenExecutionError, audit_execution
from stream.opt.solver import ConstraintSelection
from stream.stages.context import StageContext
from stream.stages.generation.generic_mapping_generation import GenericMappingGenerationStage
from stream.stages.generation.kernel_state import KernelStateStage
from stream.stages.generation.normalization_expansion import ExpandNormalizationStage
from stream.stages.generation.operator_template_compilation import OperatorTemplateCompilationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.mapping_parser import MappingParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.structural.operator_template_contract import (
    OperatorTemplate,
    OperatorTemplateAssignment,
    baseline_operator_template,
    generate_operator_template_library,
)
from stream.structural.pretiling_opportunity_census import (
    _digest,
    _parsed_compute_mapping_projection,
    _pretiling_workload_semantics,
    _shared_tensor_factors,
    tensor_relevant_signature,
)
from stream.structural.real_workload_lifting import (
    _environment_manifest,
    _file_digest,
    _run_stage,
    _source_manifest,
    _source_run_manifest,
    _unit_cost_lut,
    _validate_transfer_tiling_domains,
    load_gate2a_contract,
)
from stream.workload.node import ComputationNode, TransferNode
from stream.workload.utils import SpatialUnrollingExtentError, get_equivalent_dimension

_CONTRACT = "gate2f_posttiling_contract.json"
_FRONTEND_TRACE = ("accelerator_parser", "onnx_parser", "normalization_expansion", "generic_mapping")


class PostTilingCompatibilityError(RuntimeError):
    """The frozen Gate 2F-B relation could not be evaluated faithfully."""


def load_posttiling_compatibility_contract() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath(_CONTRACT)
    contract = json.loads(resource.read_text(encoding="utf-8"))
    _validate_contract(contract)
    return contract


def run_posttiling_compatibility(
    output_path: str | Path,
    *,
    source_commit: str | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Compare every concrete Gate 2F-A factor tuple with its production post-tiling relation."""

    contract = load_posttiling_compatibility_contract()
    source_gate, gate2fa = _load_gate2fa(contract)
    factor_specs = _factor_specs(gate2fa)
    destination = Path(output_path).resolve()
    source_before = _source_manifest(source_commit, executed_module_path=__file__)
    environment = _environment_manifest(contract)
    repeat_count = int(contract["execution"]["repeat_count"])
    shard_size = int(contract["execution"]["tuple_shard_size"])
    shard_attempt_count = sum(math.ceil(spec["total_tuple_count"] / shard_size) * repeat_count for spec in factor_specs)
    worker_count = _worker_count(max_workers, shard_attempt_count, contract)
    started = perf_counter()
    deadline = started + float(contract["execution"]["wall_time_budget_seconds"])
    with TemporaryDirectory(prefix="stream-gate2f-b-") as temporary:
        factors = _run_factor_attempts(factor_specs, Path(temporary), repeat_count, worker_count, contract, deadline)
    wall_seconds = perf_counter() - started
    source_after = _source_manifest(source_commit, executed_module_path=__file__)
    source = _source_run_manifest(source_before, source_after, destination)
    summary = _summary(factors)
    correctness = _correctness_criteria(contract, source_gate, source, environment, factors, summary, wall_seconds)
    correctness_pass = bool(correctness) and all(correctness.values())
    faithfulness = _faithfulness_criteria(contract, summary) if correctness_pass else {}
    faithfulness_pass = bool(faithfulness) and all(faithfulness.values())
    if not correctness_pass:
        verdict = contract["outcome_policy"]["correctness_failure"]
    elif faithfulness_pass:
        verdict = contract["outcome_policy"]["correctness_and_faithfulness_pass"]
    else:
        verdict = contract["outcome_policy"]["correctness_pass_relation_mismatch"]
    payload = {
        "contract": contract,
        "contract_sha256": _contract_digest(),
        "verdict": verdict,
        "run_status": "COMPLETED" if correctness_pass else "INVALID",
        "correctness_verdict": "PASS" if correctness_pass else "FAIL",
        "faithfulness_verdict": "PASS" if faithfulness_pass else "MISMATCH" if correctness_pass else "NOT_RUN",
        "evidence_class": "POSTTILING_PRODUCTION_RELATION",
        "criteria": {"correctness": correctness, "faithfulness": faithfulness},
        "source_gate": source_gate,
        "source": source,
        "environment": environment,
        "execution": {
            "factor_attempt_count": len(factor_specs) * repeat_count,
            "shard_attempt_count": shard_attempt_count,
            "tuple_shard_size": shard_size,
            "max_workers": worker_count,
            "wall_seconds": round(wall_seconds, 6),
            "tta_solve_invoked": False,
        },
        "factors": factors,
        "summary": summary,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(destination)
    return payload


def write_posttiling_compatibility_provenance(
    report_path: str | Path,
    report: dict[str, Any],
    invocation: tuple[str, ...],
    stdout_text: str,
    stderr_text: str,
) -> Path:
    """Write replay metadata and immutable output hashes beside a completed report."""

    report_path = Path(report_path).resolve()
    stdout_path = report_path.with_suffix(report_path.suffix + ".stdout.log")
    stderr_path = report_path.with_suffix(report_path.suffix + ".stderr.log")
    manifest_path = report_path.with_suffix(report_path.suffix + ".run.json")
    _atomic_write(stdout_path, stdout_text)
    _atomic_write(stderr_path, stderr_text)
    environment = report["environment"]
    manifest = {
        "schema": "stream-gate-run-provenance-v1",
        "host": environment["host"],
        "source_commit": report["source"]["commit"],
        "invocation": list(invocation),
        "report": {"path": report_path.name, "sha256": _plain_digest(report_path)},
        "stdout": {"path": stdout_path.name, "sha256": _plain_digest(stdout_path)},
        "stderr": {"path": stderr_path.name, "sha256": _plain_digest(stderr_path)},
        "environment": environment,
        "environment_sha256": _digest(environment),
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def verify_posttiling_compatibility_provenance(manifest_path: str | Path) -> bool:
    """Verify completion, artifact hashes, environment, and source identity."""

    try:
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema"] != "stream-gate-run-provenance-v1" or not manifest["source_commit"]:
            return False
        for artifact in ("report", "stdout", "stderr"):
            entry = manifest[artifact]
            relative = Path(entry["path"])
            if relative.is_absolute() or relative.parent != Path("."):
                return False
            if _plain_digest(manifest_path.parent / relative) != entry["sha256"]:
                return False
        report = json.loads((manifest_path.parent / manifest["report"]["path"]).read_text(encoding="utf-8"))
        return (
            report["run_status"] == "COMPLETED"
            and report["correctness_verdict"] == "PASS"
            and report["source"]["commit"] == manifest["source_commit"]
            and report["environment"] == manifest["environment"]
            and manifest["host"] == manifest["environment"]["host"]
            and _digest(manifest["environment"]) == manifest["environment_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _factor_specs(gate2fa: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for workload_id, result in gate2fa["workloads"].items():
        manifest = result.get("manifest")
        if not result.get("valid") or manifest is None:
            raise PostTilingCompatibilityError(f"Gate 2F-A workload {workload_id!r} has no valid manifest")
        for group_index, group in enumerate(manifest["groups"]):
            for factor_index, factor in enumerate(group["factors"]):
                specs.append(
                    {
                        "workload_id": workload_id,
                        "family": result["family"],
                        "group": group_index,
                        "factor": factor_index,
                        "factor_id": factor["id"],
                        "total_tuple_count": factor["total_tuple_count"],
                    }
                )
    return sorted(specs, key=lambda item: (item["workload_id"], item["group"], item["factor"]))


def _run_factor_attempts(specs, temporary_root, repeat_count, max_workers, contract, deadline):
    attempts: dict[str, list[dict[int, dict[str, Any]]]] = {
        _factor_key(spec): [{} for _ in range(repeat_count)] for spec in specs
    }
    shard_size = int(contract["execution"]["tuple_shard_size"])
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gate2f-b") as executor:
        pending = {}
        for spec in specs:
            for repeat in range(repeat_count):
                for start in range(0, spec["total_tuple_count"], shard_size):
                    stop = min(start + shard_size, spec["total_tuple_count"])
                    work_dir = temporary_root / f"{_factor_key(spec)}-{repeat}-{start}-{stop}"
                    future = executor.submit(
                        _isolated_factor_attempt,
                        spec,
                        work_dir,
                        repeat,
                        contract,
                        deadline,
                        start,
                        stop,
                    )
                    pending[future] = (_factor_key(spec), repeat, start)
        for future in as_completed(pending):
            factor_key, repeat, start = pending[future]
            attempts[factor_key][repeat][start] = future.result()
    return {
        _factor_key(spec): _summarize_factor(
            spec,
            [
                _merge_factor_shards(spec, repeat, shards, shard_size)
                for repeat, shards in enumerate(attempts[_factor_key(spec)])
            ],
            repeat_count,
        )
        for spec in specs
    }


def _isolated_factor_attempt(
    spec, work_dir, repeat, contract, deadline, tuple_start: int = 0, tuple_stop: int | None = None
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    result_path = work_dir / "result.json"
    request_path.write_text(
        json.dumps(
            {
                "spec": spec,
                "work_dir": str(work_dir / "preparation"),
                "result_path": str(result_path),
                "tuple_start": tuple_start,
                "tuple_stop": spec["total_tuple_count"] if tuple_stop is None else tuple_stop,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(3000 + repeat)
    started = perf_counter()
    remaining = deadline - perf_counter()
    if remaining <= 0:
        return {
            "status": "ENVIRONMENT_FAILURE",
            "repeat": repeat,
            "hash_seed": environment["PYTHONHASHSEED"],
            "wall_seconds": 0.0,
            "detail": "global Gate 2F-B deadline expired before worker start",
        }
    try:
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "stream.structural.posttiling_compatibility",
                "--worker-request",
                str(request_path),
            ),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "ENVIRONMENT_FAILURE",
            "repeat": repeat,
            "hash_seed": environment["PYTHONHASHSEED"],
            "wall_seconds": round(perf_counter() - started, 6),
            "detail": "worker terminated at the global Gate 2F-B deadline",
        }
    if completed.returncode or not result_path.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip() or "worker produced no result"
        return {
            "status": "ENVIRONMENT_FAILURE",
            "repeat": repeat,
            "hash_seed": environment["PYTHONHASHSEED"],
            "wall_seconds": round(perf_counter() - started, 6),
            "detail": detail[-4000:],
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "repeat": repeat,
            "hash_seed": environment["PYTHONHASHSEED"],
            "wall_seconds": round(perf_counter() - started, 6),
        }
    )
    return result


def _merge_factor_shards(spec, repeat, shards, shard_size) -> dict[str, Any]:  # noqa: PLR0911
    """Reconstruct one deterministic full-factor attempt from isolated ordinal shards."""

    expected_ranges = [
        (start, min(start + shard_size, spec["total_tuple_count"]))
        for start in range(0, spec["total_tuple_count"], shard_size)
    ]
    if sorted(shards) != [start for start, _ in expected_ranges]:
        return {"status": "ENVIRONMENT_FAILURE", "repeat": repeat, "detail": "one or more shards never started"}
    ordered = [shards[start] for start, _ in expected_ranges]
    failed = next((item for item in ordered if item.get("status") != "VALID"), None)
    if failed is not None:
        return {
            "status": failed["status"],
            "repeat": repeat,
            "hash_seed": failed.get("hash_seed"),
            "wall_seconds": sum(float(item.get("wall_seconds", 0.0)) for item in ordered),
            "detail": failed.get("detail", "factor shard failed"),
        }
    manifests = [item["manifest"] for item in ordered]
    expected_seed = str(3000 + repeat)
    if any(item.get("hash_seed") != expected_seed for item in ordered):
        return {"status": "CORRECTNESS_FAILURE", "repeat": repeat, "detail": "factor shard hash-seed drift"}
    stable_fields = (
        "spec",
        "frontend_trace",
        "mapping_parser_reference_match",
        "domain_hashes_match",
        "factor_manifest_match",
    )
    if any(manifest[field] != manifests[0][field] for manifest in manifests[1:] for field in stable_fields):
        return {"status": "CORRECTNESS_FAILURE", "repeat": repeat, "detail": "factor shard manifest drift"}
    if any(
        (manifest["tuple_start"], manifest["tuple_stop"]) != expected_range
        for manifest, expected_range in zip(manifests, expected_ranges, strict=True)
    ):
        return {"status": "CORRECTNESS_FAILURE", "repeat": repeat, "detail": "factor shard range drift"}
    if any(
        len(manifest["rows"]) != stop - start or len(manifest["tuple_key_lines"]) != stop - start
        for manifest, (start, stop) in zip(manifests, expected_ranges, strict=True)
    ):
        return {"status": "CORRECTNESS_FAILURE", "repeat": repeat, "detail": "factor shard cardinality drift"}
    rows = [row for manifest in manifests for row in manifest["rows"]]
    if [row["ordinal"] for row in rows] != list(range(spec["total_tuple_count"])):
        return {"status": "CORRECTNESS_FAILURE", "repeat": repeat, "detail": "factor shard row conservation failed"}
    tuple_digest = sha256()
    for manifest in manifests:
        for line in manifest["tuple_key_lines"]:
            tuple_digest.update(line.encode() + b"\n")
    witnesses: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        for witness_id, witness in manifest["witnesses"].items():
            if witness_id in witnesses and witnesses[witness_id] != witness:
                return {"status": "CORRECTNESS_FAILURE", "repeat": repeat, "detail": "witness hash collision"}
            witnesses[witness_id] = witness
    boundary = {
        event.value: sum(manifest["execution_boundary"][event.value] for manifest in manifests)
        for event in ExecutionEvent
    }
    merged = {field: manifests[0][field] for field in stable_fields}
    merged.update(
        {
            "tuple_count": spec["total_tuple_count"],
            "tuple_key_sha256": tuple_digest.hexdigest(),
            "witnesses": witnesses,
            "rows": rows,
            "execution_boundary": boundary,
        }
    )
    return {
        "status": "VALID",
        "manifest": merged,
        "repeat": repeat,
        "hash_seed": expected_seed,
        "aggregate_shard_wall_seconds": sum(float(item["wall_seconds"]) for item in ordered),
    }


def _run_worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = Path(request["result_path"])
    try:
        with audit_execution(forbidden=frozenset(ExecutionEvent)) as execution:
            manifest = _evaluate_factor(
                request["spec"],
                Path(request["work_dir"]),
                tuple_start=int(request["tuple_start"]),
                tuple_stop=int(request["tuple_stop"]),
            )
        manifest["execution_boundary"] = execution.manifest()
        result = {"status": "VALID", "manifest": manifest}
    except ForbiddenExecutionError as error:
        result = {"status": "CORRECTNESS_FAILURE", "detail": str(error)}
    except Exception as error:  # noqa: BLE001 - preserve the exact fail-closed worker failure
        result = {"status": "CORRECTNESS_FAILURE", "detail": f"{type(error).__name__}: {error}"}
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


def _evaluate_factor(  # noqa: PLR0915
    spec: dict[str, Any], work_dir: Path, *, tuple_start: int = 0, tuple_stop: int | None = None
) -> dict[str, Any]:
    contract = load_posttiling_compatibility_contract()
    _, gate2fa = _load_gate2fa(contract)
    gate2a = load_gate2a_contract()
    workload_spec = next(item for item in gate2a["workloads"] if item["id"] == spec["workload_id"])
    expected_workload = gate2fa["workloads"][spec["workload_id"]]
    expected_group = expected_workload["manifest"]["groups"][spec["group"]]
    expected_factor = expected_group["factors"][spec["factor"]]
    if expected_factor["id"] != spec["factor_id"] or expected_factor["total_tuple_count"] != spec["total_tuple_count"]:
        raise PostTilingCompatibilityError(f"factor spec drift for {_factor_key(spec)}")

    context, mapping_path, frontend_trace = _mapping_parser_context(workload_spec, spec["group"], work_dir)
    workload = context.require_value("workload", "Gate2F-B")
    mapping = context.require_value("mapping", "Gate2F-B")
    _verify_mapping_seam(workload, mapping, Path(mapping_path), expected_group)
    library = generate_operator_template_library(workload, mapping)
    baseline_templates = tuple(
        baseline_operator_template(workload, mapping, node) for node in workload.get_computation_nodes()
    )
    baseline_mapping = _parsed_compute_mapping_projection(workload, mapping)
    source_graph = (tuple(workload.nodes), tuple(workload.edges))
    domains = {
        node.name: tuple(template for template in library.templates if template.target == node.name)
        for node in workload.get_computation_nodes()
    }
    _verify_template_domains(domains, expected_group)
    current_factors = _shared_tensor_factors(
        workload,
        domains,
        int(contract["execution"]["maximum_direct_factor_tuples"]),
    )
    if current_factors != expected_group["factors"]:
        raise PostTilingCompatibilityError(f"factor semantics drift for {_factor_key(spec)}")
    consumers = [_node_by_name(workload, identity.split(":", 1)[1]) for identity in expected_factor["consumers"]]
    tensor = next((item for item in workload.tensors if item.name == expected_factor["tensor"]), None)
    if tensor is None or any(tensor not in consumer.inputs for consumer in consumers):
        raise PostTilingCompatibilityError(f"factor tensor identity drift for {_factor_key(spec)}")
    template_domains = tuple(domains[consumer.name] for consumer in consumers)
    tuple_count = math.prod(len(domain) for domain in template_domains)
    if tuple_count != expected_factor["total_tuple_count"]:
        raise PostTilingCompatibilityError(f"factor tuple denominator drift for {_factor_key(spec)}")
    tuple_stop = tuple_count if tuple_stop is None else tuple_stop
    if not 0 <= tuple_start < tuple_stop <= tuple_count:
        raise PostTilingCompatibilityError(f"invalid tuple shard [{tuple_start}, {tuple_stop})")

    tuple_key_lines = []
    witnesses: dict[str, dict[str, Any]] = {}
    rows = []
    tuple_work = work_dir / "tuples"
    tuple_work.mkdir(parents=True, exist_ok=True)
    for ordinal in range(tuple_start, tuple_stop):
        selected = _cartesian_tuple_at(template_domains, ordinal)
        selected_by_target = {template.target: template for template in selected}
        expected_templates = tuple(selected_by_target.get(item.target, item) for item in baseline_templates)
        indices = tuple(template_domains[index].index(template) for index, template in enumerate(selected))
        tuple_key = [template.key for template in selected]
        tuple_key_lines.append(json.dumps(tuple_key, separators=(",", ":")))
        pre_signatures = tuple(
            tensor_relevant_signature(workload, consumer, tensor, template)
            for consumer, template in zip(consumers, selected, strict=True)
        )
        pre = len({signature for signature in pre_signatures if signature}) <= 1
        try:
            outcome = _evaluate_tuple(
                workload,
                mapping,
                context.require_value("accelerator", "Gate2F-B"),
                library,
                selected,
                expected_templates,
                baseline_mapping,
                expected_factor,
                tuple_work,
                contract,
            )
            witness = outcome.pop("witness")
            witness_id = _digest(witness)
            witnesses.setdefault(witness_id, witness)
            row = {
                "ordinal": ordinal,
                "template_indices": list(indices),
                "status": "VALID",
                "pre": pre,
                "post": outcome["post"],
                "post_relation_witness": outcome["post_relation_witness"],
                "literal_survival": outcome["literal_survival"],
                "lineage_witness": outcome["lineage_witness"],
                "nonempty_post_domains": outcome["nonempty_post_domains"],
                "ssis_semantics": outcome["ssis_semantics"],
                "literal_failure_stages": outcome["literal_failure_stages"],
                "stage_trace": outcome["stage_trace"],
                "witness_id": witness_id,
            }
        except Exception as error:  # noqa: BLE001 - every concrete tuple receives a terminal status
            row = {
                "ordinal": ordinal,
                "template_indices": list(indices),
                "status": "INVALID",
                "pre": pre,
                "post": None,
                "detail": f"{type(error).__name__}: {error}"[:4000],
            }
        if (tuple(workload.nodes), tuple(workload.edges)) != source_graph or not _computation_mapping_matches(
            workload, mapping, baseline_templates, baseline_mapping
        ):
            row = {
                "ordinal": ordinal,
                "template_indices": list(indices),
                "status": "INVALID",
                "pre": pre,
                "post": None,
                "detail": "source workload or MappingParser mapping mutated across tuple evaluation",
            }
        rows.append(row)
    return {
        "spec": spec,
        "frontend_trace": frontend_trace,
        "mapping_parser_reference_match": True,
        "domain_hashes_match": True,
        "factor_manifest_match": True,
        "tuple_count": tuple_count,
        "tuple_start": tuple_start,
        "tuple_stop": tuple_stop,
        "tuple_key_lines": tuple_key_lines,
        "witnesses": witnesses,
        "rows": rows,
    }


def _mapping_parser_context(workload_spec, group_index: int, work_dir: Path):
    hardware_path = Path(load_gate2a_contract()["hardware"])
    context = StageContext.from_kwargs(
        accelerator=str(hardware_path),
        workload_path=workload_spec["path"],
        output_path=str(work_dir),
        intra_core_tiling=None,
        fusion_cut_points=None,
    )
    trace = []
    for stage, label in (
        (AcceleratorParserStage, "accelerator_parser"),
        (ONNXModelParserStage, "onnx_parser"),
        (ExpandNormalizationStage, "normalization_expansion"),
        (GenericMappingGenerationStage, "generic_mapping"),
    ):
        context = _run_stage(stage, context)
        trace.append(label)
    if tuple(trace) != _FRONTEND_TRACE:
        raise PostTilingCompatibilityError(f"frontend trace drifted: {trace}")
    sub_workloads = context.require_value("sub_workloads", "Gate2F-B")
    mapping_paths = context.require_value("group_mapping_paths", "Gate2F-B")
    if group_index >= len(sub_workloads) or len(sub_workloads) != len(mapping_paths):
        raise PostTilingCompatibilityError("fusion-group denominator drifted")
    group_context = StageContext.from_kwargs(
        accelerator=context.require_value("accelerator", "Gate2F-B"),
        workload=sub_workloads[group_index],
        mapping_path=mapping_paths[group_index],
        output_path=str(work_dir / f"group_{group_index}"),
    )
    return _run_stage(MappingParserStage, group_context), mapping_paths[group_index], trace


def _verify_mapping_seam(workload, mapping, mapping_path: Path, expected_group: dict[str, Any]) -> None:
    observed = {
        "operator_ids": [f"ComputationNode:{node.name}" for node in workload.get_computation_nodes()],
        "workload_semantics": _pretiling_workload_semantics(workload),
        "parsed_compute_mapping": _parsed_compute_mapping_projection(workload, mapping),
        "mapping_file_sha256": _file_digest(mapping_path),
    }
    expected = {
        "operator_ids": expected_group["operator_ids"],
        "workload_semantics": expected_group["workload_semantics"],
        "parsed_compute_mapping": expected_group["parsed_compute_mapping"],
        "mapping_file_sha256": expected_group["mapping_file_sha256"],
    }
    if observed != expected:
        raise PostTilingCompatibilityError("MappingParser-seam reference drift")


def _verify_template_domains(domains, expected_group) -> None:
    expected = {operator["name"]: operator for operator in expected_group["operators"]}
    if set(domains) != set(expected):
        raise PostTilingCompatibilityError("operator-template denominator drift")
    for name, domain in domains.items():
        keys = [template.key for template in domain]
        if keys != expected[name]["templates"] or _digest(keys) != expected[name]["domain_sha256"]:
            raise PostTilingCompatibilityError(f"operator-template domain drift for {name}")


def _evaluate_tuple(  # noqa: PLR0913, PLR0915
    workload,
    mapping,
    accelerator,
    library,
    selected,
    expected_templates,
    baseline_mapping,
    factor,
    output_path,
    contract,
):
    trace = []
    literal_checks = []

    def mapping_check(stage, current_workload, current_mapping, *, normalized=False, updated_workload=False):
        expected_fused_groups = (
            _translated_fused_group_projection(workload, mapping, current_workload)
            if updated_workload
            else baseline_mapping["fused_groups"]
        )
        passed = _computation_mapping_matches(
            current_workload,
            current_mapping,
            expected_templates,
            baseline_mapping,
            expected_fused_groups=expected_fused_groups,
            normalized=normalized,
        )
        literal_checks.append({"stage": stage, "passed": passed})
        return passed

    assignment = OperatorTemplateAssignment("gate2f-b", tuple(selected))
    context = StageContext.from_kwargs(
        workload=workload,
        mapping=mapping,
        accelerator=accelerator,
        output_path=str(output_path),
        operator_template_assignment=assignment,
        operator_template_library=library,
        visualize_tiled_workload=bool(contract["execution"]["visualize_tiled_workload"]),
    )
    context = _run_stage(OperatorTemplateCompilationStage, context)
    trace.append("operator_template_compilation")
    literal_survival = mapping_check("operator_template_compilation", context.get("workload"), context.get("mapping"))
    context = _run_stage(KernelStateStage, context)
    trace.append("kernel_state")
    literal_survival &= mapping_check("kernel_state", context.get("workload"), context.get("mapping"))
    context = _run_stage(TilingGenerationStage, context)
    trace.append("tiling_generation")
    literal_survival &= mapping_check("tiling_generation", context.get("workload"), context.get("mapping"))
    tiled_workload = context.require_value("workload", "Gate2F-B")
    tiled_mapping = context.require_value("mapping", "Gate2F-B")
    scheduler = SteadyStateScheduler(
        tiled_workload,
        accelerator,
        tiled_mapping,
        context.require_value("fusion_splits", "Gate2F-B"),
        _unit_cost_lut(tiled_workload, tiled_mapping),
        nb_cols_to_use=int(contract["execution"]["nb_cols_to_use"]),
        backend=contract["execution"]["backend"],
        constraint_selection=ConstraintSelection(),
        max_transfer_plans_per_endpoint=int(contract["execution"]["max_transfer_plans_per_endpoint"]),
    )
    scheduler.ssw = scheduler.build_transfer_graph()
    trace.append("transfer_graph")
    literal_survival &= mapping_check("transfer_graph", scheduler.ssw, scheduler.mapping)
    scheduler.fusion_splits = scheduler.update_fusion_splits()
    trace.append("fusion_splits")
    scheduler.normalize_computation_mapping()
    trace.append("computation_mapping_normalization")
    literal_survival &= mapping_check(
        "computation_mapping_normalization", scheduler.ssw, scheduler.mapping, normalized=True
    )
    try:
        scheduler.mapping = scheduler.update_transfer_mapping()
    except (SharedInputTilingIncompatibilityError, TransferDomainIncompatibilityError) as error:
        trace.append(
            "transfer_domain_rejected"
            if isinstance(error, TransferDomainIncompatibilityError)
            else "transfer_mapping_rejected"
        )
        decisions = _factor_decisions(scheduler.tensor_relevant_tiling_decisions, factor)
        if error.decision not in decisions or not decisions or error.decision.accepted:
            raise PostTilingCompatibilityError(
                "structured incompatibility witness did not match the frozen factor"
            ) from error
        return {
            "post": False,
            "post_relation_witness": True,
            "literal_survival": literal_survival,
            "lineage_witness": True,
            "nonempty_post_domains": None,
            "ssis_semantics": None,
            "literal_failure_stages": [item["stage"] for item in literal_checks if not item["passed"]],
            "stage_trace": trace,
            "witness": {"outcome": "REJECTED", "decisions": [_decision_manifest(item) for item in decisions]},
        }
    trace.append("transfer_mapping")
    literal_survival &= mapping_check(
        "transfer_mapping", scheduler.ssw, scheduler.mapping, normalized=True, updated_workload=True
    )
    scheduler.cost_lut = scheduler.update_cost_lut()
    trace.append("cost_lut")
    try:
        scheduler.ssis = scheduler.generate_ssis()
    except SpatialUnrollingExtentError as error:
        decisions = _factor_decisions(scheduler.tensor_relevant_tiling_decisions, factor)
        if not _factor_explains_ssis_rejection(decisions, error):
            raise PostTilingCompatibilityError(
                "SSIS spatial-unrolling rejection did not match the frozen factor"
            ) from error
        trace.append("ssis_rejected")
        return {
            "post": False,
            "post_relation_witness": True,
            "literal_survival": literal_survival,
            "lineage_witness": True,
            "nonempty_post_domains": True,
            "ssis_semantics": None,
            "literal_failure_stages": [item["stage"] for item in literal_checks if not item["passed"]],
            "stage_trace": trace,
            "witness": {
                "outcome": "REJECTED",
                "reason": {
                    "kind": "spatial_unrolling_extent",
                    "dimension": str(error.dimension),
                    "dimension_size": error.dimension_size,
                    "spatial_unrolling": error.spatial_unrolling,
                    "source_nodes": list(error.source_nodes),
                },
                "decisions": [_decision_manifest(item) for item in decisions],
            },
        }
    trace.append("ssis")
    literal_survival &= mapping_check("ssis", scheduler.ssw, scheduler.mapping, normalized=True, updated_workload=True)
    _validate_transfer_tiling_domains(scheduler)
    decisions = _factor_decisions(scheduler.tensor_relevant_tiling_decisions, factor)
    completed = [decision for decision in decisions if decision.role == "completed_transfer_mapping"]
    lineage_transfers = [
        transfer for transfer, lineage in scheduler.transfer_lineage.items() if _lineage_matches(lineage, factor)
    ]
    lineage_witness = bool(completed) and {item.transfer for item in completed} == {
        transfer.name for transfer in lineage_transfers
    }
    nonempty = bool(completed) and all(
        item.result_resource_option_count > 0 and item.result_memory_option_count > 0 for item in completed
    )
    if any(not decision.accepted for decision in decisions):
        raise PostTilingCompatibilityError("completed transfer mapping retained a rejected compatibility decision")
    if any(transfer not in scheduler.ssis for transfer in lineage_transfers):
        raise PostTilingCompatibilityError("a factor transfer is missing from the generated SSIS")
    ssis_semantics = _ssis_semantics_valid(scheduler, lineage_transfers)
    return {
        "post": True,
        "post_relation_witness": lineage_witness and nonempty and ssis_semantics,
        "literal_survival": literal_survival,
        "lineage_witness": lineage_witness,
        "nonempty_post_domains": nonempty,
        "ssis_semantics": ssis_semantics,
        "literal_failure_stages": [item["stage"] for item in literal_checks if not item["passed"]],
        "stage_trace": trace,
        "witness": {
            "outcome": "ACCEPTED",
            "decisions": [_decision_manifest(item) for item in decisions],
            "ssis": [
                _ssis_entry(scheduler, transfer) for transfer in sorted(lineage_transfers, key=lambda item: item.name)
            ],
        },
    }


def _selected_templates_match(workload, mapping, selected: tuple[OperatorTemplate, ...], *, normalized=False) -> bool:
    for template in selected:
        node = _node_by_name(workload, template.target)
        node_mapping = mapping.get(node)
        if len(node_mapping.resource_allocation) != 1:
            return False
        if tuple(core.id for core in node_mapping.resource_allocation[0]) != template.core_ids:
            return False
        tiling = node_mapping.inter_core_tiling[0] if len(node_mapping.inter_core_tiling) == 1 else ()
        if normalized:
            dims = workload.get_dims(node)
            expected = tuple((dims[position], factor) for position, factor in template.splits)
            if tiling != expected:
                return False
        elif tuple((dimension.position, factor) for dimension, factor in tiling if factor > 1) != template.splits:
            return False
    return True


def _translated_fused_group_projection(source_workload, source_mapping, current_workload) -> list[dict[str, Any]]:
    """Project MappingParser fused-group literals into the current workload's dimension coordinates."""

    return [
        {
            "name": group.name,
            "layers": list(group.layers),
            "intra_core_tiling": [
                {
                    "position": get_equivalent_dimension(source_workload, current_workload, dimension).position,
                    "factor": factor,
                }
                for dimension, factor in group.intra_core_tiling
            ],
        }
        for group in source_mapping.fused_groups
    ]


def _computation_mapping_matches(
    workload,
    mapping,
    expected_templates,
    baseline_mapping,
    *,
    expected_fused_groups=None,
    normalized=False,
):
    if not _selected_templates_match(workload, mapping, expected_templates, normalized=normalized):
        return False
    observed = _parsed_compute_mapping_projection(workload, mapping)
    if (
        observed["fused_groups"]
        != (baseline_mapping["fused_groups"] if expected_fused_groups is None else expected_fused_groups)
        or observed["runtime_args"] != baseline_mapping["runtime_args"]
    ):
        return False
    for name, baseline in baseline_mapping["nodes"].items():
        current = observed["nodes"].get(name)
        if (
            current is None
            or current["memory_options"] != baseline["memory_options"]
            or current["kernel"] != baseline["kernel"]
        ):
            return False
    return True


def _ssis_semantics_valid(scheduler, transfers) -> bool:
    for transfer in transfers:
        variables = scheduler.ssis[transfer].variables
        if not variables or any(variable.size < 1 for variable in variables):
            return False
        spatial = {
            (variable.dimension, variable.size)
            for variable in variables
            if variable.type.name == "SPATIAL" and variable.effect.name == "VARYING"
        }
        for tiling in scheduler.mapping.get(transfer).inter_core_tiling:
            if any((dimension, factor) not in spatial for dimension, factor in tiling):
                return False
    return True


def _factor_decisions(decisions: list[TensorRelevantTilingDecision], factor) -> list[TensorRelevantTilingDecision]:
    matching = [decision for decision in decisions if _lineage_matches(decision.lineage, factor)]
    if not matching:
        raise PostTilingCompatibilityError("production emitted no compatibility witness for the frozen factor")
    return matching


def _factor_explains_ssis_rejection(
    decisions: list[TensorRelevantTilingDecision], error: SpatialUnrollingExtentError
) -> bool:
    """Whether a frozen factor both requested and concretized the rejected unrolling."""

    completed = [decision for decision in decisions if decision.role == "completed_transfer_mapping"]
    matching = [
        decision
        for decision in completed
        if any(
            node in error.source_nodes and (error.dimension, error.spatial_unrolling) in projection
            for node, projection in decision.projections
        )
    ]
    return bool(error.source_nodes) and bool(matching) and all(
        decision.accepted and decision.result_resource_option_count > 0 and decision.result_memory_option_count > 0
        for decision in matching
    )


def _cartesian_tuple_at(domains, ordinal: int):
    """Return one itertools.product tuple by mixed-radix ordinal without scanning its prefix."""

    if ordinal < 0:
        raise IndexError("Cartesian ordinal must be non-negative")
    indices = [0] * len(domains)
    remainder = ordinal
    for index in range(len(domains) - 1, -1, -1):
        remainder, indices[index] = divmod(remainder, len(domains[index]))
    if remainder:
        raise IndexError("Cartesian ordinal is outside the domain product")
    return tuple(domain[index] for domain, index in zip(domains, indices, strict=True))


def _lineage_matches(lineage: TransferLineage, factor: dict[str, Any]) -> bool:
    observed_consumers = dict(zip(lineage.consumers, lineage.consumer_operand_indices, strict=True))
    expected_consumers = dict(zip(factor["consumers"], map(tuple, factor["consumer_operand_indices"]), strict=True))
    return (
        lineage.tensor == factor["tensor"]
        and lineage.producer == factor["producer"]
        and observed_consumers == expected_consumers
    )


def _decision_manifest(decision: TensorRelevantTilingDecision) -> dict[str, Any]:
    def tiling_manifest(tiling):
        return [{"dimension": str(dimension), "factor": factor} for dimension, factor in tiling]

    return {
        "transfer": decision.transfer,
        "lineage": asdict(decision.lineage),
        "projections": [{"node": node, "tiling": tiling_manifest(tiling)} for node, tiling in decision.projections],
        "selected_reference": decision.selected_reference,
        "role": decision.role,
        "accepted": decision.accepted,
        "compatible_projection_set": decision.compatible,
        "result_tilings": [tiling_manifest(tiling) for tiling in decision.result_tilings],
        "result_resource_option_count": decision.result_resource_option_count,
        "result_memory_option_count": decision.result_memory_option_count,
        "rejection_reason": decision.rejection_reason,
    }


def _ssis_entry(scheduler, transfer: TransferNode) -> dict[str, Any]:
    return {
        "transfer": transfer.name,
        "variables": [
            {
                "dimension": str(variable.dimension),
                "size": variable.size,
                "effect": variable.effect.name,
                "type": variable.type.name,
                "reuse": variable.reuse.name,
            }
            for variable in scheduler.ssis[transfer].variables
        ],
    }


def _node_by_name(workload, name: str) -> ComputationNode:
    nodes = [node for node in workload.get_computation_nodes() if node.name == name]
    if len(nodes) != 1:
        raise PostTilingCompatibilityError(f"computation node {name!r} is not unique")
    return nodes[0]


def _summarize_factor(spec, attempts, repeat_count):
    valid = [attempt for attempt in attempts if attempt.get("status") == "VALID"]
    hashes = [_digest(attempt["manifest"]) for attempt in valid]
    deterministic = len(valid) == repeat_count and len(set(hashes)) == 1
    manifest = valid[0]["manifest"] if deterministic else None
    compact_attempts = []
    for attempt in attempts:
        row = {key: value for key, value in attempt.items() if key != "manifest"}
        if attempt.get("status") == "VALID":
            row["semantic_hash"] = _digest(attempt["manifest"])
        compact_attempts.append(row)
    return {
        "spec": spec,
        "valid": deterministic,
        "deterministic": deterministic,
        "semantic_hash": hashes[0] if deterministic else None,
        "manifest": manifest,
        "attempts": compact_attempts,
    }


def _summary(factors):
    manifests = [result["manifest"] for result in factors.values() if result["manifest"]]
    rows = [row for manifest in manifests for row in manifest["rows"]]
    valid_rows = [row for row in rows if row["status"] == "VALID"]
    accepted_rows = [row for row in valid_rows if row["post"]]
    rejected_rows = [row for row in valid_rows if not row["post"]]
    return {
        "factor_count": len(factors),
        "valid_factor_count": sum(result["valid"] for result in factors.values()),
        "expected_tuple_count": sum(result["spec"]["total_tuple_count"] for result in factors.values()),
        "enumerated_tuple_count": len(rows),
        "valid_tuple_count": len(valid_rows),
        "invalid_tuple_count": len(rows) - len(valid_rows),
        "accepted_tuple_count": len(accepted_rows),
        "rejected_tuple_count": len(rejected_rows),
        "true_positive_count": sum(row["pre"] and row["post"] for row in valid_rows),
        "true_negative_count": sum(not row["pre"] and not row["post"] for row in valid_rows),
        "false_positive_count": sum(row["pre"] and not row["post"] for row in valid_rows),
        "false_negative_count": sum(not row["pre"] and row["post"] for row in valid_rows),
        "literal_survival_count": sum(row["literal_survival"] for row in valid_rows),
        "lineage_witness_count": sum(row["lineage_witness"] for row in valid_rows),
        "accepted_nonempty_post_domain_count": sum(row["nonempty_post_domains"] for row in accepted_rows),
        "accepted_ssis_semantics_count": sum(row["ssis_semantics"] for row in accepted_rows),
        "rejected_relation_witness_count": sum(row["post_relation_witness"] for row in rejected_rows),
        "forbidden_execution_events": sum(sum(manifest["execution_boundary"].values()) for manifest in manifests),
    }


def _correctness_criteria(contract, source_gate, source, environment, factors, summary, wall_seconds):
    manifests = [result["manifest"] for result in factors.values() if result["manifest"]]
    valid_count = summary["valid_tuple_count"]
    accepted_count = summary["accepted_tuple_count"]
    rejected_count = summary["rejected_tuple_count"]
    expected_count = summary["expected_tuple_count"]
    observations = {
        "source_identified": source["identified"],
        "source_gate_artifacts_match": source_gate["artifacts_match"],
        "environment_compatible": environment["compatible"],
        "gate2fa_environment_match": environment == source_gate["environment"],
        "factor_denominator_match": len(factors) == source_gate["factor_count"],
        "concrete_tuple_conservation": summary["enumerated_tuple_count"] == expected_count,
        "valid_tuple_ratio": valid_count / expected_count if expected_count else 0.0,
        "deterministic_factor_repeat_ratio": sum(result["deterministic"] for result in factors.values()) / len(factors),
        "stage_allowlist_exact": bool(manifests)
        and all(
            tuple(row.get("stage_trace", ()))
            in (
                tuple(contract["allowed_stage_trace"]["compatible"]),
                tuple(contract["allowed_stage_trace"]["incompatible"]),
                tuple(contract["allowed_stage_trace"]["incompatible_transfer_domain"]),
                tuple(contract["allowed_stage_trace"]["incompatible_ssis"]),
            )
            for manifest in manifests
            for row in manifest["rows"]
            if row["status"] == "VALID"
        ),
        "literal_survival_ratio": summary["literal_survival_count"] / valid_count if valid_count else 0.0,
        "lineage_witness_ratio": summary["lineage_witness_count"] / valid_count if valid_count else 0.0,
        "accepted_nonempty_post_domain_ratio": (
            summary["accepted_nonempty_post_domain_count"] / accepted_count if accepted_count else 1.0
        ),
        "accepted_ssis_semantics_ratio": (
            summary["accepted_ssis_semantics_count"] / accepted_count if accepted_count else 1.0
        ),
        "rejected_relation_witness_ratio": (
            summary["rejected_relation_witness_count"] / rejected_count if rejected_count else 1.0
        ),
        "forbidden_execution_events": summary["forbidden_execution_events"],
    }
    if wall_seconds > contract["execution"]["wall_time_budget_seconds"]:
        observations["valid_tuple_ratio"] = -1.0
    return _evaluate_criteria(observations, contract["correctness_criteria"])


def _faithfulness_criteria(contract, summary):
    observations = {
        "false_positive_count": summary["false_positive_count"],
        "false_negative_count": summary["false_negative_count"],
    }
    return _evaluate_criteria(observations, contract["faithfulness_criteria"])


def _load_gate2fa(contract):
    expected = contract["source_gate"]
    report_path = Path(expected["report_artifact"])
    provenance_path = Path(expected["provenance_artifact"])
    repeat_path = Path(expected["repeat_audit_artifact"])
    encoded = report_path.read_text(encoding="utf-8")
    report_bytes = gzip.decompress(base64.b64decode(encoded))
    report = json.loads(report_bytes)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
    matches = (
        _file_digest(report_path) == expected["report_artifact_sha256"]
        and _file_digest(provenance_path) == expected["provenance_artifact_sha256"]
        and _file_digest(repeat_path) == expected["repeat_audit_artifact_sha256"]
        and sha256(report_bytes).hexdigest() == provenance["report"]["sha256"]
        and report.get("verdict") == expected["required_verdict"]
        and report.get("correctness_verdict") == expected["required_correctness_verdict"]
        and report.get("run_status") == "COMPLETED"
        and repeat.get("semantic_comparison", {}).get("equal") is True
    )
    if not matches:
        raise PostTilingCompatibilityError("Gate 2F-A source artifacts do not match the frozen contract")
    return {
        "artifacts_match": True,
        "environment": report["environment"],
        "factor_count": report["summary"]["shared_tensor_factor_count"],
        "source_commit": report["source"]["commit"],
    }, report


def _evaluate_criteria(observations, expected):
    if set(observations) != set(expected):
        raise PostTilingCompatibilityError("contract criteria do not match observations")
    return {
        key: observations[key] >= target
        if isinstance(target, int | float) and not isinstance(target, bool)
        else observations[key] == target
        for key, target in expected.items()
    }


def _worker_count(requested, attempts, contract):
    if requested is not None and requested < 1:
        raise ValueError("max_workers must be at least one")
    limit = requested if requested is not None else int(contract["execution"]["max_workers"])
    return min(attempts, os.cpu_count() or 1, limit)


def _factor_key(spec) -> str:
    return f"{spec['workload_id']}-g{spec['group']}-f{spec['factor']}"


def _contract_digest() -> str:
    resource = files("stream.structural.contracts").joinpath(_CONTRACT)
    return _file_digest(Path(str(resource)))


def _plain_digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _validate_contract(contract):
    expected = {
        "contract",
        "version",
        "source_gate",
        "relation",
        "allowed_stage_trace",
        "execution",
        "environment",
        "correctness_criteria",
        "faithfulness_criteria",
        "outcome_policy",
        "excluded_claims",
    }
    if set(contract) != expected or contract["version"] != "gate2f-b-v1":
        raise PostTilingCompatibilityError("invalid Gate 2F-B contract schema")
    if contract["execution"]["run_tta"] or contract["execution"]["run_structural_search"]:
        raise PostTilingCompatibilityError("Gate 2F-B must remain prepare-only")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-request", type=Path, required=True)
    args = parser.parse_args()
    _run_worker(args.worker_request)


if __name__ == "__main__":
    _main()


__all__ = [
    "PostTilingCompatibilityError",
    "load_posttiling_compatibility_contract",
    "run_posttiling_compatibility",
    "verify_posttiling_compatibility_provenance",
    "write_posttiling_compatibility_provenance",
]
