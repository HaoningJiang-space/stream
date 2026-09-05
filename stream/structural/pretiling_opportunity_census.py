"""Gate 2F-A: real-workload opportunity census at the pre-tiling compiler seam."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, TypeAlias

from stream.execution_boundary import ExecutionEvent, ForbiddenExecutionError, audit_execution
from stream.stages.context import StageContext
from stream.stages.generation.generic_mapping_generation import GenericMappingGenerationStage
from stream.stages.generation.normalization_expansion import ExpandNormalizationStage
from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
from stream.stages.parsing.mapping_parser import MappingParserStage
from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage
from stream.structural.operator_template_contract import (
    OperatorTemplate,
    OperatorTemplateAssignment,
    baseline_operator_template,
    compile_operator_templates,
    generate_operator_template_library,
)
from stream.structural.real_workload_lifting import (
    _environment_manifest,
    _file_digest,
    _run_stage,
    _source_manifest,
    _source_run_manifest,
    load_gate2a_contract,
)
from stream.workload.affine_access import map_dim_positions
from stream.workload.iterator_type import IteratorType, derive_iterator_types
from stream.workload.node import ComputationNode, HasInputs, HasIterationSpace, HasOutputs, TransferNode
from stream.workload.tensor import Tensor

Signature: TypeAlias = tuple[tuple[str, int], ...]

_FRONTEND_TRACE = ("accelerator_parser", "onnx_parser", "normalization_expansion", "generic_mapping")
_GROUP_TRACE = ("mapping_parser",)
_CONTRACT = "gate2f_pretiling_contract.json"
_MIN_FACTOR_ARITY = 2


class PreTilingOpportunityError(RuntimeError):
    """The frozen Gate 2F-A census cannot be evaluated faithfully."""


def load_pretiling_opportunity_contract() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath(_CONTRACT)
    contract = json.loads(resource.read_text(encoding="utf-8"))
    _validate_contract(contract)
    return contract


def run_pretiling_opportunity_census(
    output_path: str | Path,
    *,
    source_commit: str | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run the frozen prepare-only census without KernelState, tiling, scheduler preparation, or TTA."""

    contract = load_pretiling_opportunity_contract()
    source_gates, accepted_denominator = _load_source_gates(contract)
    workload_specs = load_gate2a_contract()["workloads"]
    destination = Path(output_path).resolve()
    source_before = _source_manifest(source_commit, executed_module_path=__file__)
    environment = _environment_manifest(contract)
    repeat_count = int(contract["execution"]["repeat_count"])
    worker_count = _worker_count(max_workers, len(workload_specs) * repeat_count, contract)
    started = perf_counter()
    with TemporaryDirectory(prefix="stream-gate2f-a-") as temporary:
        workloads = _run_attempts(
            workload_specs,
            accepted_denominator,
            Path(temporary),
            repeat_count,
            worker_count,
            contract,
        )
    wall_seconds = perf_counter() - started
    source_after = _source_manifest(source_commit, executed_module_path=__file__)
    source = _source_run_manifest(source_before, source_after, destination)
    summary = _summary(workloads)
    correctness = _correctness_criteria(contract, source_gates, source, environment, workloads, summary, wall_seconds)
    opportunity = _opportunity_criteria(contract, workloads, summary) if all(correctness.values()) else {}
    correctness_pass = bool(correctness) and all(correctness.values())
    opportunity_pass = bool(opportunity) and all(opportunity.values())
    if not correctness_pass:
        verdict = contract["outcome_policy"]["correctness_failure"]
    elif opportunity_pass:
        verdict = contract["outcome_policy"]["correctness_and_opportunity_pass"]
    else:
        verdict = contract["outcome_policy"]["correctness_pass_opportunity_negative"]
    payload = {
        "contract": contract,
        "contract_sha256": _contract_digest(),
        "verdict": verdict,
        "run_status": "COMPLETED" if correctness_pass else "INVALID",
        "correctness_verdict": "PASS" if correctness_pass else "FAIL",
        "opportunity_verdict": "PASS" if opportunity_pass else "NEGATIVE" if correctness_pass else "NOT_RUN",
        "evidence_class": "PRETILING_POTENTIAL_ONLY",
        "criteria": {"correctness": correctness, "opportunity": opportunity},
        "source_gates": source_gates,
        "source": source,
        "environment": environment,
        "execution": {
            "attempt_count": len(workload_specs) * repeat_count,
            "max_workers": worker_count,
            "wall_seconds": round(wall_seconds, 6),
        },
        "workloads": workloads,
        "summary": summary,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(destination)
    return payload


def write_pretiling_opportunity_provenance(
    report_path: str | Path,
    report: dict[str, Any],
    invocation: tuple[str, ...],
    stdout_text: str,
    stderr_text: str,
) -> Path:
    """Write a replay manifest and immutable log hashes beside a completed report."""

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


def verify_pretiling_opportunity_provenance(manifest_path: str | Path) -> bool:
    """Verify the completion marker, artifact hashes, and environment digest."""

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
        report_path = manifest_path.parent / manifest["report"]["path"]
        report = json.loads(report_path.read_text(encoding="utf-8"))
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


def tensor_relevant_signature(
    workload,
    consumer: ComputationNode,
    tensor: Tensor,
    template: OperatorTemplate,
) -> Signature:
    """Project a pre-tiling template onto dimensions indexing ``tensor``.

    This is a structural proxy. Gate 2F-A does not claim equivalence to the
    scheduler's post-transformation transfer projection on real workloads.
    """

    relevant_positions = map_dim_positions(consumer.get_mapping(tensor))
    dimensions = workload.get_dims(consumer)
    return tuple(
        (str(dimensions[position]), factor) for position, factor in template.splits if position in relevant_positions
    )


def count_signature_compatibility(
    multiplicities: tuple[dict[Signature, int], ...],
) -> dict[str, int | bool]:
    """Count empty-or-equal signature tuples with exact integer arithmetic."""

    invalid_domains = any(not counts or any(value < 1 for value in counts.values()) for counts in multiplicities)
    if len(multiplicities) < _MIN_FACTOR_ARITY or invalid_domains:
        raise ValueError("compatibility counting requires at least two non-empty positive multiplicity domains")
    total = math.prod(sum(counts.values()) for counts in multiplicities)
    compatible = _compatible_signature_count(multiplicities)
    empty = math.prod(counts.get((), 0) for counts in multiplicities)
    support_sizes = []
    for index, counts in enumerate(multiplicities):
        remaining = multiplicities[:index] + multiplicities[index + 1 :]
        supported = counts.get((), 0) if _compatible_signature_count(remaining) > 0 else 0
        for signature, count in counts.items():
            if signature and all(
                other == index or multiplicities[other].get((), 0) + multiplicities[other].get(signature, 0) > 0
                for other in range(len(multiplicities))
            ):
                supported += count
        support_sizes.append(supported)
    projected_product = math.prod(support_sizes)
    return {
        "total_tuple_count": total,
        "all_empty_tuple_count": empty,
        "compatible_tuple_count": compatible,
        "unary_projection_product": projected_product,
        "noncartesian": compatible < projected_product,
    }


def _compatible_signature_count(multiplicities: tuple[dict[Signature, int], ...]) -> int:
    if not multiplicities:
        return 1
    if len(multiplicities) == 1:
        return sum(multiplicities[0].values())
    empty = math.prod(counts.get((), 0) for counts in multiplicities)
    signatures = set().union(*(set(counts) - {()} for counts in multiplicities))
    return empty + sum(
        math.prod(counts.get((), 0) + counts.get(signature, 0) for counts in multiplicities) - empty
        for signature in signatures
    )


def _run_attempts(workload_specs, accepted_denominator, temporary_root, repeat_count, max_workers, contract):
    attempts: dict[str, list[dict[str, Any] | None]] = {spec["id"]: [None] * repeat_count for spec in workload_specs}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gate2f-a") as executor:
        pending = {}
        for spec in workload_specs:
            for repeat in range(repeat_count):
                work_dir = temporary_root / f"{spec['id']}-{repeat}"
                future = executor.submit(
                    _isolated_attempt, spec, accepted_denominator[spec["id"]], work_dir, repeat, contract
                )
                pending[future] = (spec["id"], repeat)
        for future in as_completed(pending):
            workload_id, repeat = pending[future]
            attempts[workload_id][repeat] = future.result()
    return {
        spec["id"]: _summarize_attempts(spec, [item for item in attempts[spec["id"]] if item is not None], repeat_count)
        for spec in workload_specs
    }


def _isolated_attempt(workload_spec, accepted_denominator, work_dir, repeat, contract) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    result_path = work_dir / "result.json"
    request_path.write_text(
        json.dumps(
            {
                "workload_spec": workload_spec,
                "accepted_denominator": accepted_denominator,
                "work_dir": str(work_dir / "preparation"),
                "result_path": str(result_path),
                "maximum_operator_domain_size": contract["execution"]["maximum_operator_domain_size"],
                "maximum_direct_factor_tuples": contract["execution"]["maximum_direct_factor_tuples"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(2000 + repeat)
    started = perf_counter()
    completed = subprocess.run(
        (sys.executable, "-m", "stream.structural.pretiling_opportunity_census", "--worker-request", str(request_path)),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode or not result_path.is_file():
        detail = completed.stderr.strip() or completed.stdout.strip() or "worker produced no result"
        return {
            "status": "ENVIRONMENT_FAILURE",
            "repeat": repeat,
            "hash_seed": environment["PYTHONHASHSEED"],
            "wall_seconds": round(perf_counter() - started, 6),
            "detail": detail[-2000:],
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


def _run_worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    result_path = Path(request["result_path"])
    try:
        with audit_execution(forbidden=frozenset(ExecutionEvent)) as execution:
            manifest = _prepare_once(
                request["workload_spec"],
                request["accepted_denominator"],
                Path(request["work_dir"]),
                int(request["maximum_operator_domain_size"]),
                int(request["maximum_direct_factor_tuples"]),
            )
        manifest["execution_boundary"] = execution.manifest()
        result = {"status": "VALID", "manifest": manifest}
    except ForbiddenExecutionError as error:
        result = {"status": "CORRECTNESS_FAILURE", "detail": str(error)}
    except Exception as error:  # noqa: BLE001 - preserve exact worker failure in the Gate artifact
        result = {"status": "CORRECTNESS_FAILURE", "detail": f"{type(error).__name__}: {error}"}
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


def _prepare_once(workload_spec, accepted_denominator, work_dir, maximum_domain_size, maximum_direct_tuples):
    work_dir.mkdir(parents=True, exist_ok=True)
    hardware_path = Path(load_gate2a_contract()["hardware"])
    context = StageContext.from_kwargs(
        accelerator=str(hardware_path),
        workload_path=workload_spec["path"],
        output_path=str(work_dir),
        intra_core_tiling=None,
        fusion_cut_points=None,
    )
    frontend_trace = []
    for stage, label in (
        (AcceleratorParserStage, "accelerator_parser"),
        (ONNXModelParserStage, "onnx_parser"),
        (ExpandNormalizationStage, "normalization_expansion"),
        (GenericMappingGenerationStage, "generic_mapping"),
    ):
        context = _run_stage(stage, context)
        frontend_trace.append(label)
    groups = []
    for index, (workload, mapping_path) in enumerate(
        zip(context.get("sub_workloads"), context.get("group_mapping_paths"), strict=True)
    ):
        group_context = StageContext.from_kwargs(
            accelerator=context.get("accelerator"),
            workload=workload,
            mapping_path=mapping_path,
            output_path=str(work_dir / f"group_{index}"),
        )
        group_context = _run_stage(MappingParserStage, group_context)
        groups.append(
            _group_manifest(
                index,
                workload,
                group_context.get("mapping"),
                _file_digest(Path(mapping_path)),
                maximum_domain_size,
                maximum_direct_tuples,
            )
        )
    current_denominator = [group["operator_ids"] for group in groups]
    workload_digest = _file_digest(Path(workload_spec["path"]))
    return {
        "workload_id": workload_spec["id"],
        "family": workload_spec["family"],
        "path": workload_spec["path"],
        "sha256": workload_digest,
        "frontend_trace": frontend_trace,
        "group_trace": list(_GROUP_TRACE),
        "group_count": len(groups),
        "operator_count": sum(group["operator_count"] for group in groups),
        "denominator_matches_gate2a": (current_denominator == accepted_denominator["operator_ids"]),
        "inputs_match_gate2a": workload_digest == accepted_denominator["sha256"]
        and _file_digest(hardware_path) == accepted_denominator["hardware_sha256"],
        "pretiling_reference_match": pretiling_mapping_reference_matches(
            groups,
            accepted_denominator["pretiling_groups"],
        ),
        "groups": groups,
    }


def _group_manifest(index, workload, mapping, mapping_file_sha256, maximum_domain_size, maximum_direct_tuples):
    library = generate_operator_template_library(workload, mapping)
    domains = {
        node.name: tuple(template for template in library.templates if template.target == node.name)
        for node in workload.get_computation_nodes()
    }
    operators = []
    generated_compile = True
    reduction_splits = 0
    for node in workload.get_computation_nodes():
        domain = domains[node.name]
        if len(domain) > maximum_domain_size:
            raise PreTilingOpportunityError(f"{node.name}: domain size {len(domain)} exceeds {maximum_domain_size}")
        baseline = baseline_operator_template(workload, mapping, node)
        iterator_types = derive_iterator_types(node)
        for ordinal, template in enumerate(domain):
            reduction_splits += sum(
                iterator_types[position] is not IteratorType.PARALLEL for position, _ in template.splits
            )
            compiled = compile_operator_templates(
                workload,
                mapping,
                OperatorTemplateAssignment(f"compile-check-{node.name}-{ordinal}", (template,)),
                library,
            )
            compiled_mapping = compiled.mapping.get(node)
            compiled_splits = (
                tuple((dimension.position, factor) for dimension, factor in compiled_mapping.inter_core_tiling[0])
                if compiled_mapping.inter_core_tiling
                else ()
            )
            generated_compile &= (
                tuple(core.id for core in compiled_mapping.resource_allocation[0]) == template.core_ids
                and compiled_splits == template.splits
            )
        operators.append(
            {
                "id": f"ComputationNode:{node.name}",
                "name": node.name,
                "operation": str(node.type),
                "parsed_core_pool": [core.id for core in mapping.get(node).resource_allocation[0]],
                "baseline_template": baseline.key,
                "baseline_retained": baseline in domain,
                "domain_size": len(domain),
                "domain_sha256": _digest([template.key for template in domain]),
                "templates": [template.key for template in domain],
            }
        )
    shared_tensor_candidates = sum(
        len([node for node in workload.get_computation_nodes() if tensor in node.inputs]) > 1
        for tensor in workload.tensors
    )
    factors = _shared_tensor_factors(workload, domains, maximum_direct_tuples)
    state_space_log10 = sum(math.log10(len(domain)) for domain in domains.values())
    return {
        "group": index,
        "stage_trace": list(_GROUP_TRACE),
        "parsed_compute_mapping": _parsed_compute_mapping_projection(workload, mapping),
        "workload_semantics": _pretiling_workload_semantics(workload),
        "mapping_file_sha256": mapping_file_sha256,
        "operator_ids": [operator["id"] for operator in operators],
        "operator_count": len(operators),
        "state_sum": sum(operator["domain_size"] for operator in operators),
        "nondegenerate_operator_count": sum(operator["domain_size"] > 1 for operator in operators),
        "nominal_space_log10": state_space_log10,
        "generated_templates_compile": generated_compile,
        "reduction_axis_split_count": reduction_splits,
        "operators": operators,
        "shared_tensor_candidate_count": shared_tensor_candidates,
        "shared_tensor_factor_count": len(factors),
        "noncartesian_factor_count": sum(factor["noncartesian"] for factor in factors),
        "factors": factors,
    }


def _shared_tensor_factors(workload, domains, maximum_direct_tuples):
    factors = []
    order = {node: index for index, node in enumerate(workload.dataflow_sort())}
    for tensor in workload.tensors:
        consumers = sorted(
            (node for node in workload.get_computation_nodes() if tensor in node.inputs),
            key=order.__getitem__,
        )
        if len(consumers) < _MIN_FACTOR_ARITY:
            continue
        producers = [node for node in workload.nodes if isinstance(node, HasOutputs) and tensor in node.outputs]
        if len(producers) != 1:
            raise PreTilingOpportunityError(f"tensor {tensor.name!r} does not have one producer")
        counters = tuple(
            Counter(
                tensor_relevant_signature(workload, consumer, tensor, template) for template in domains[consumer.name]
            )
            for consumer in consumers
        )
        counts = count_signature_compatibility(tuple(dict(counter) for counter in counters))
        if counts["total_tuple_count"] > maximum_direct_tuples:
            raise PreTilingOpportunityError(
                f"tensor {tensor.name!r} direct audit exceeds {maximum_direct_tuples} tuples"
            )
        direct = _direct_compatibility_count(counters)
        factors.append(
            {
                "id": f"{type(producers[0]).__name__}:{producers[0].name}->{tensor.name}",
                "tensor": tensor.name,
                "producer": f"{type(producers[0]).__name__}:{producers[0].name}",
                "consumers": [f"ComputationNode:{consumer.name}" for consumer in consumers],
                "consumer_operand_indices": [
                    [index for index, operand in enumerate(consumer.inputs) if operand is tensor]
                    for consumer in consumers
                ],
                "arity": len(consumers),
                "domain_sizes": [sum(counter.values()) for counter in counters],
                "signature_multiplicities": [
                    [
                        {"signature": _signature_manifest(signature), "count": count}
                        for signature, count in sorted(counter.items(), key=lambda item: item[0])
                    ]
                    for counter in counters
                ],
                **counts,
                "direct_compatible_tuple_count": direct,
                "analytic_direct_match": direct == counts["compatible_tuple_count"],
                "compatible_ratio": counts["compatible_tuple_count"] / counts["total_tuple_count"],
            }
        )
    return factors


def _direct_compatibility_count(counters: tuple[Counter[Signature], ...]) -> int:
    total = 0
    for signatures in product(*(tuple(counter) for counter in counters)):
        if len({signature for signature in signatures if signature}) <= 1:
            total += math.prod(counter[signature] for counter, signature in zip(counters, signatures, strict=True))
    return total


def _signature_manifest(signature: Signature) -> list[dict[str, Any]]:
    return [{"dimension": dimension, "factor": factor} for dimension, factor in signature]


def _parsed_compute_mapping_projection(workload, mapping) -> dict[str, Any]:
    nodes = {}
    for node in sorted(workload.get_computation_nodes(), key=lambda item: item.name):
        node_mapping = mapping.get(node)
        nodes[node.name] = {
            "resource_options": [[core.id for core in option] for option in node_mapping.resource_allocation],
            "tiling_options": [
                [{"position": dimension.position, "factor": factor} for dimension, factor in option]
                for option in node_mapping.inter_core_tiling
            ],
            "memory_options": [[core.id for core in option] for option in node_mapping.memory_allocation],
            "kernel": type(node_mapping.kernel).__name__ if node_mapping.kernel is not None else None,
        }
    return {
        "nodes": nodes,
        "fused_groups": [
            {
                "name": group.name,
                "layers": list(group.layers),
                "intra_core_tiling": [
                    {"position": dimension.position, "factor": factor}
                    for dimension, factor in group.intra_core_tiling
                ],
            }
            for group in mapping.fused_groups
        ],
        "runtime_args": dict(sorted(mapping.runtime_args.items())),
    }


def pretiling_mapping_reference_matches(groups, reference_groups) -> bool:
    """Compare complete MappingParser-seam workload, mapping, and mapping-file identity."""

    return len(groups) == len(reference_groups) and all(
        group["operator_ids"] == [f"ComputationNode:{name}" for name in reference["operator_ids"]]
        and group["workload_semantics"] == reference["workload_semantics"]
        and group["parsed_compute_mapping"] == reference["mapping"]
        and group["mapping_file_sha256"] == reference["mapping_file_sha256"]
        for group, reference in zip(groups, reference_groups, strict=True)
    )


def _pretiling_workload_semantics(workload) -> dict[str, Any]:
    """Serialize the graph and affine semantics consumed by the pre-tiling census."""

    dimension_sizes = workload.get_dimension_sizes()
    nodes = []
    for node in workload.dataflow_sort():
        row: dict[str, Any] = {"id": f"{type(node).__name__}:{node.name}", "node_class": type(node).__name__}
        if isinstance(node, HasInputs):
            row["inputs"] = [
                {"name": tensor.name, "shape": list(tensor.shape), "type": str(tensor.operand_type)}
                for tensor in node.inputs
            ]
        if isinstance(node, HasOutputs):
            row["outputs"] = [
                {"name": tensor.name, "shape": list(tensor.shape), "type": str(tensor.operand_type)}
                for tensor in node.outputs
            ]
        if isinstance(node, HasIterationSpace):
            row["dimensions"] = [
                {"global_position": position, "size": dimension_sizes[position]}
                for position in workload.global_idxs[node]
            ]
            row["operand_maps"] = [str(mapping) for mapping in node.operand_mapping]
        if isinstance(node, ComputationNode):
            row["operation"] = str(node.type)
            row["fused_kernel"] = node.fused_kernel
            row["reduction_axes"] = list(getattr(node, "reduction_axes", ()))
        if isinstance(node, TransferNode):
            row["transfer_type"] = node.transfer_type.name
        if hasattr(node, "op_type"):
            row["fusion_op_type"] = node.op_type
        nodes.append(row)
    edges = []
    for source, target in workload.edges:
        shared = (
            [tensor.name for tensor in source.outputs if tensor in target.inputs]
            if isinstance(source, HasOutputs) and isinstance(target, HasInputs)
            else []
        )
        edges.append(
            {
                "source": f"{type(source).__name__}:{source.name}",
                "target": f"{type(target).__name__}:{target.name}",
                "tensors": shared,
            }
        )
    return {"nodes": nodes, "edges": edges, "dimension_sizes": list(dimension_sizes)}


def _summarize_attempts(spec, attempts, repeat_count):
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
        "family": spec["family"],
        "path": spec["path"],
        "sha256": _file_digest(Path(spec["path"])),
        "valid": deterministic,
        "deterministic": deterministic,
        "semantic_hash": hashes[0] if deterministic else None,
        "manifest": manifest,
        "attempts": compact_attempts,
    }


def _summary(workloads):
    manifests = [result["manifest"] for result in workloads.values() if result["manifest"]]
    groups = [group for manifest in manifests for group in manifest["groups"]]
    operators = [operator for group in groups for operator in group["operators"]]
    factors = [factor for group in groups for factor in group["factors"]]
    return {
        "required_workloads": len(workloads),
        "valid_workloads": sum(result["valid"] for result in workloads.values()),
        "operator_count": len(operators),
        "candidate_state_count_sum": sum(operator["domain_size"] for operator in operators),
        "nondegenerate_operator_count": sum(operator["domain_size"] > 1 for operator in operators),
        "operator_nondegenerate_ratio": (
            sum(operator["domain_size"] > 1 for operator in operators) / len(operators) if operators else 0.0
        ),
        "candidate_state_count_min": min((operator["domain_size"] for operator in operators), default=0),
        "candidate_state_count_max": max((operator["domain_size"] for operator in operators), default=0),
        "shared_tensor_factor_count": len(factors),
        "noncartesian_factor_count": sum(factor["noncartesian"] for factor in factors),
        "noncartesian_workload_families": sorted(
            {
                result["family"]
                for result in workloads.values()
                if result["manifest"]
                and any(group["noncartesian_factor_count"] for group in result["manifest"]["groups"])
            }
        ),
        "maximum_group_nominal_space_log10": max((group["nominal_space_log10"] for group in groups), default=0.0),
    }


def _correctness_criteria(contract, source_gates, source, environment, workloads, summary, wall_seconds):
    manifests = [result["manifest"] for result in workloads.values() if result["manifest"]]
    groups = [group for manifest in manifests for group in manifest["groups"]]
    operators = [operator for group in groups for operator in group["operators"]]
    factors = [factor for group in groups for factor in group["factors"]]
    denominator = {entry["id"]: entry for entry in contract["workload_denominator"]}
    denominator_match = workload_denominator_matches(workloads, denominator)
    gate2a_inputs_match = bool(manifests) and all(manifest["inputs_match_gate2a"] for manifest in manifests)
    gate2a = source_gates["gate2a"]
    observations: dict[str, Any] = {
        "source_identified": source["identified"],
        "source_gate_hashes_match": all(
            gate["hash_matches"] and gate.get("reference_hash_matches", True) for gate in source_gates.values()
        ),
        "environment_compatible": environment["compatible"],
        "gate2a_recorded_environment_match": environment == gate2a["environment"],
        "gate2a_pretiling_reference_match": bool(manifests)
        and all(manifest["pretiling_reference_match"] for manifest in manifests),
        "workload_denominator_match": denominator_match,
        "gate2a_inputs_match": gate2a_inputs_match,
        "preparation_success_ratio": summary["valid_workloads"] / summary["required_workloads"],
        "deterministic_repeat_ratio": sum(result["deterministic"] for result in workloads.values()) / len(workloads),
        "stage_allowlist_exact": bool(manifests)
        and all(
            manifest["frontend_trace"] == contract["allowed_stage_trace"]["frontend"]
            and all(
                group["stage_trace"] == contract["allowed_stage_trace"]["per_group"] for group in manifest["groups"]
            )
            for manifest in manifests
        ),
        "forbidden_execution_events": sum(sum(manifest["execution_boundary"].values()) for manifest in manifests),
        "candidate_templates_unique": bool(operators)
        and all(operator["domain_size"] == len(set(operator["templates"])) for operator in operators),
        "generated_templates_compile": bool(groups) and all(group["generated_templates_compile"] for group in groups),
        "baseline_retention_ratio": (
            sum(operator["baseline_retained"] for operator in operators) / len(operators) if operators else 0.0
        ),
        "reduction_axis_split_count": sum(group["reduction_axis_split_count"] for group in groups),
        "shared_tensor_factor_coverage": (
            sum(group["shared_tensor_factor_count"] for group in groups)
            / sum(group["shared_tensor_candidate_count"] for group in groups)
            if groups and sum(group["shared_tensor_candidate_count"] for group in groups)
            else 1.0
        ),
        "analytic_direct_factor_count_match": bool(factors)
        and all(factor["analytic_direct_match"] for factor in factors),
        "maximum_domain_size_within_budget": summary["candidate_state_count_max"]
        <= contract["execution"]["maximum_operator_domain_size"],
        "wall_time_within_budget": wall_seconds <= contract["execution"]["wall_time_budget_seconds"],
    }
    return _evaluate_criteria(observations, contract["correctness_criteria"])


def _opportunity_criteria(contract, workloads, summary):
    observations = {
        "operator_nondegenerate_ratio": summary["operator_nondegenerate_ratio"],
        "minimum_noncartesian_workload_families": len(summary["noncartesian_workload_families"]),
    }
    return _evaluate_criteria(observations, contract["opportunity_criteria"])


def workload_denominator_matches(workloads, denominator) -> bool:
    """Require exact workload family, group, operator, and Gate 2A identity retention."""

    return set(workloads) == set(denominator) and all(
        result["family"] == denominator[workload_id]["family"]
        and result["manifest"] is not None
        and result["manifest"]["group_count"] == denominator[workload_id]["group_count"]
        and result["manifest"]["operator_count"] == denominator[workload_id]["operator_count"]
        and result["manifest"]["denominator_matches_gate2a"]
        for workload_id, result in workloads.items()
    )


def _evaluate_criteria(observations, expected):
    if set(observations) != set(expected):
        raise PreTilingOpportunityError("contract criteria do not match observations")
    return {
        key: observations[key] >= target
        if isinstance(target, int | float) and not isinstance(target, bool)
        else observations[key] == target
        for key, target in expected.items()
    }


def _load_source_gates(contract):
    loaded = {}
    accepted_denominator = None
    for name, expected in contract["source_gates"].items():
        path = Path(expected["artifact"])
        digest = _file_digest(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        matches = digest == expected["sha256"] and payload.get("verdict") == expected["required_verdict"]
        if not matches:
            raise PreTilingOpportunityError(f"source Gate {name} does not match the frozen contract")
        loaded[name] = {
            "path": str(path),
            "sha256": digest,
            "verdict": payload["verdict"],
            "hash_matches": True,
        }
        if name == "gate2a":
            reference, reference_digest = _load_pretiling_reference(expected["pretiling_reference"], payload)
            loaded[name].update(
                {
                    "source_commit": payload["source"]["commit"],
                    "environment": payload["environment"],
                    "reference_path": expected["pretiling_reference"]["artifact"],
                    "reference_sha256": reference_digest,
                    "reference_hash_matches": True,
                }
            )
            accepted_denominator = {
                workload_id: {
                    "sha256": result["sha256"],
                    "hardware_sha256": payload["hardware"]["sha256"],
                    "pretiling_groups": reference["workloads"][workload_id]["groups"],
                    "operator_ids": [
                        [
                            node["id"]
                            for node in group["input_workload"]["nodes"]
                            if node["id"].startswith("ComputationNode:")
                        ]
                        for group in result["manifest"]["groups"]
                    ],
                }
                for workload_id, result in payload["workloads"].items()
            }
    if accepted_denominator is None:
        raise PreTilingOpportunityError("Gate 2A source denominator is unavailable")
    return loaded, accepted_denominator


def _load_pretiling_reference(expected, gate2a):
    path = Path(expected["artifact"])
    digest = _file_digest(path)
    encoded = path.read_text(encoding="utf-8")
    reference = json.loads(gzip.decompress(base64.b64decode(encoded)).decode("utf-8"))
    environment = gate2a["environment"]
    recorded_environment = {
        "python_version": environment["python_version"],
        "packages": environment["packages"],
    }
    accepted_workloads = {
        workload_id: {
            "family": result["family"],
            "sha256": result["sha256"],
            "operator_ids": [
                [
                    node["id"].split(":", 1)[1]
                    for node in group["input_workload"]["nodes"]
                    if node["id"].startswith("ComputationNode:")
                ]
                for group in result["manifest"]["groups"]
            ],
        }
        for workload_id, result in gate2a["workloads"].items()
    }
    reference_workloads = {
        workload_id: {
            "family": result["family"],
            "sha256": result["sha256"],
            "operator_ids": [group["operator_ids"] for group in result["groups"]],
        }
        for workload_id, result in reference.get("workloads", {}).items()
    }
    valid = (
        digest == expected["sha256"]
        and reference.get("schema") == expected["schema"]
        and reference.get("source") == {"commit": gate2a["source"]["commit"], "clean": True}
        and reference.get("instrument", {}).get("commit") == expected["instrument_commit"]
        and reference.get("instrument", {}).get("path") == expected["instrument_path"]
        and reference.get("instrument", {}).get("sha256")
        == _git_blob_digest(expected["instrument_commit"], expected["instrument_path"])
        and reference.get("environment") == recorded_environment
        and reference.get("hardware") == {
            "path": gate2a["hardware"]["path"],
            "sha256": gate2a["hardware"]["sha256"],
        }
        and reference_workloads == accepted_workloads
    )
    if not valid:
        raise PreTilingOpportunityError("Gate 2A pre-tiling reference does not match its frozen provenance")
    return reference, digest


def _git_blob_digest(commit: str, path: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "show", f"{commit}:{path}"),
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return "sha256:" + sha256(result.stdout).hexdigest()


def _worker_count(requested, attempts, contract):
    if requested is not None and requested < 1:
        raise ValueError("max_workers must be at least one")
    limit = requested if requested is not None else int(contract["execution"]["max_workers"])
    return min(attempts, os.cpu_count() or 1, limit)


def _contract_digest() -> str:
    resource = files("stream.structural.contracts").joinpath(_CONTRACT)
    return _file_digest(Path(str(resource)))


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
        "design_status",
        "threshold_origin",
        "source_gates",
        "workload_denominator",
        "allowed_stage_trace",
        "template_rules",
        "signature_proxy",
        "execution",
        "environment",
        "correctness_criteria",
        "opportunity_criteria",
        "outcome_policy",
        "excluded_claims",
    }
    if set(contract) != expected or contract["version"] != "gate2f-a-v1":
        raise PreTilingOpportunityError("invalid Gate 2F-A contract schema")
    denominator = contract["workload_denominator"]
    if not denominator or len({entry["id"] for entry in denominator}) != len(denominator):
        raise PreTilingOpportunityError("workload denominator must be non-empty and unique")
    if tuple(contract["allowed_stage_trace"]["frontend"]) != _FRONTEND_TRACE:
        raise PreTilingOpportunityError("frontend stage allowlist drifted")
    if tuple(contract["allowed_stage_trace"]["per_group"]) != _GROUP_TRACE:
        raise PreTilingOpportunityError("group stage allowlist drifted")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-request", type=Path, required=True)
    args = parser.parse_args()
    _run_worker(args.worker_request)


if __name__ == "__main__":
    _main()


__all__ = [
    "PreTilingOpportunityError",
    "count_signature_compatibility",
    "load_pretiling_opportunity_contract",
    "run_pretiling_opportunity_census",
    "tensor_relevant_signature",
    "verify_pretiling_opportunity_provenance",
    "workload_denominator_matches",
    "write_pretiling_opportunity_provenance",
]
