"""Gate 2B opportunity census over an accepted Gate 2A lifting artifact."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any

_MIN_ADJACENT_TRANSFERS = 2


class OpportunityCensusError(RuntimeError):
    """The accepted lifting artifact cannot support an exact opportunity census."""


def load_gate2b_contract() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath("gate2b_contract.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def run_opportunity_census(
    output_path: str | Path,
    *,
    source_report: str | Path | None = None,
) -> dict[str, Any]:
    """Measure executable placement/path opportunity without running optimization or TTA."""

    contract = load_gate2b_contract()
    source_path = Path(source_report or contract["source_gate"]["artifact"])
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_manifest = _validate_source_gate(source_path, source_payload, contract)
    workload_results = {
        workload_id: _workload_census(workload_id, source_payload["workloads"][workload_id])
        for workload_id in contract["source_gate"]["required_workloads"]
    }
    summary = _portfolio_summary(workload_results)
    verdict, opportunity_class, limitations = _classify_opportunity(summary)
    payload = {
        "contract": contract,
        "verdict": verdict,
        "opportunity_class": opportunity_class,
        "limitations": limitations,
        "source_gate": source_manifest,
        "workloads": workload_results,
        "summary": summary,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def _validate_source_gate(source_path: Path, payload: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["source_gate"]
    actual_digest = _file_digest(source_path)
    if actual_digest != expected["sha256"]:
        raise OpportunityCensusError(f"Gate 2A artifact digest mismatch: {actual_digest}")
    if payload.get("verdict") != expected["required_verdict"]:
        raise OpportunityCensusError(f"Gate 2A verdict is {payload.get('verdict')!r}")
    source_contract = payload.get("contract", {})
    if source_contract.get("contract") != expected["name"] or source_contract.get("version") != expected["version"]:
        raise OpportunityCensusError("Gate 2A contract identity mismatch")
    if not payload.get("criteria") or not all(payload["criteria"].values()):
        raise OpportunityCensusError("Gate 2A criteria are incomplete or non-passing")
    required_workloads = expected["required_workloads"]
    if set(payload.get("workloads", {})) != set(required_workloads):
        raise OpportunityCensusError("Gate 2A workload denominator mismatch")
    return {
        "path": str(source_path),
        "sha256": actual_digest,
        "verdict": payload["verdict"],
        "source_commit": payload["source"]["commit"],
        "workload_semantic_hashes": {
            workload_id: payload["workloads"][workload_id]["semantic_hash"] for workload_id in required_workloads
        },
    }


def _workload_census(workload_id: str, workload_result: dict[str, Any]) -> dict[str, Any]:
    manifest = workload_result.get("manifest")
    if not workload_result.get("valid") or manifest is None:
        raise OpportunityCensusError(f"{workload_id}: Gate 2A workload is not valid")
    tensors = []
    for group in manifest["groups"]:
        group_index = int(group["group"])
        type_widths = _tensor_type_widths(group["steady_state_workload"])
        tensor_rows = {row["id"]: row for row in group["domains"]["tensors"]}
        for staging in group["domains"]["staging_tensors"]:
            tensor_id = staging["tensor"]
            if tensor_id not in tensor_rows or tensor_id not in type_widths:
                raise OpportunityCensusError(f"{workload_id}/group-{group_index}: missing tensor {tensor_id!r}")
            tensors.append(
                _tensor_opportunity(
                    workload_id,
                    group_index,
                    staging,
                    tensor_rows[tensor_id],
                    type_widths[tensor_id],
                )
            )
    variable_ids = [tensor["variable_id"] for tensor in tensors]
    if len(variable_ids) != len(set(variable_ids)):
        raise OpportunityCensusError(f"{workload_id}: duplicate group-qualified tensor variable")
    return {
        "family": workload_result["family"],
        "semantic_hash": workload_result["semantic_hash"],
        "opportunity": _summarize_tensors(tensors),
        "tensors": tensors,
    }


def _tensor_type_widths(workload_manifest: dict[str, Any]) -> dict[str, int]:
    widths: dict[str, int] = {}
    for node in workload_manifest["nodes"]:
        for tensor in (*node.get("inputs", ()), *node.get("outputs", ())):
            width = _fixed_bitwidth(tensor["type"])
            previous = widths.setdefault(tensor["name"], width)
            if previous != width:
                raise OpportunityCensusError(f"tensor {tensor['name']!r} has inconsistent element widths")
    return widths


def _fixed_bitwidth(type_name: str) -> int:
    match = re.fullmatch(r"[A-Za-z]+(\d+)", type_name)
    if match is None or int(match.group(1)) < 1:
        raise OpportunityCensusError(f"unsupported fixed-bitwidth type {type_name!r}")
    return int(match.group(1))


def _tensor_opportunity(
    workload_id: str,
    group_index: int,
    staging: dict[str, Any],
    tensor: dict[str, Any],
    bitwidth: int,
) -> dict[str, Any]:
    choices = staging["choices"]
    if not choices:
        raise OpportunityCensusError(f"{workload_id}/{staging['tensor']}: empty assignment domain")
    assignment_counts = []
    for choice in choices:
        path_counts = choice["compatible_path_counts"]
        if not path_counts or any(int(count) < 1 for count in path_counts):
            raise OpportunityCensusError(f"{workload_id}/{staging['tensor']}: empty compatible path domain")
        assignment_count = math.prod(int(count) for count in path_counts)
        if assignment_count != int(choice["assignment_count"]):
            raise OpportunityCensusError(f"{workload_id}/{staging['tensor']}: inconsistent assignment count")
        assignment_counts.append(assignment_count)
    domain_size = sum(assignment_counts)
    if domain_size != int(staging["assignment_count"]):
        raise OpportunityCensusError(f"{workload_id}/{staging['tensor']}: inconsistent tensor domain size")
    shape = tuple(int(extent) for extent in tensor["shape"])
    if not shape or any(extent < 1 for extent in shape):
        raise OpportunityCensusError(f"{workload_id}/{staging['tensor']}: invalid dense tensor shape")
    logical_bits = math.prod(shape) * bitwidth
    adjacent_transfer_count = len(tensor["adjacent_transfers"])
    if adjacent_transfer_count < _MIN_ADJACENT_TRANSFERS:
        raise OpportunityCensusError(f"{workload_id}/{staging['tensor']}: tensor is outside the frozen selector")
    return {
        "variable_id": f"group-{group_index}:{staging['tensor']}",
        "tensor": staging["tensor"],
        "group": group_index,
        "provenance": tensor["provenance"],
        "shape": list(shape),
        "element_bitwidth": bitwidth,
        "logical_bits": logical_bits,
        "logical_edge_bits": logical_bits * adjacent_transfer_count,
        "adjacent_transfer_count": adjacent_transfer_count,
        "baseline_placement_count": int(staging["placement_count"]),
        "compatible_placement_count": len(choices),
        "path_tuple_counts_by_placement": assignment_counts,
        "assignment_domain_size": domain_size,
        "nondegenerate": domain_size > 1,
        "placement_nondegenerate": len(choices) > 1,
        "path_nondegenerate": any(count > 1 for count in assignment_counts),
    }


def _summarize_tensors(tensors: list[dict[str, Any]]) -> dict[str, Any]:
    nondegenerate = [tensor for tensor in tensors if tensor["nondegenerate"]]
    eligible_bits = sum(tensor["logical_bits"] for tensor in tensors)
    controlled_bits = sum(tensor["logical_bits"] for tensor in nondegenerate)
    eligible_edge_bits = sum(tensor["logical_edge_bits"] for tensor in tensors)
    controlled_edge_bits = sum(tensor["logical_edge_bits"] for tensor in nondegenerate)
    domain_sizes = [tensor["assignment_domain_size"] for tensor in tensors]
    naive_space = math.prod(domain_sizes)
    return {
        "eligible_tensor_count": len(tensors),
        "nondegenerate_tensor_count": len(nondegenerate),
        "nondegenerate_tensor_ratio": len(nondegenerate) / len(tensors) if tensors else 0.0,
        "placement_nondegenerate_count": sum(tensor["placement_nondegenerate"] for tensor in tensors),
        "path_nondegenerate_count": sum(tensor["path_nondegenerate"] for tensor in tensors),
        "domain_size_histogram": dict(sorted(Counter(map(str, domain_sizes)).items())),
        "naive_assignment_space": str(naive_space),
        "naive_assignment_space_log10": math.log10(naive_space) if naive_space else None,
        "eligible_logical_bits": eligible_bits,
        "controlled_logical_bits": controlled_bits,
        "controlled_logical_bits_ratio": controlled_bits / eligible_bits if eligible_bits else 0.0,
        "eligible_logical_edge_bits": eligible_edge_bits,
        "controlled_logical_edge_bits": controlled_edge_bits,
        "controlled_logical_edge_bits_ratio": controlled_edge_bits / eligible_edge_bits if eligible_edge_bits else 0.0,
    }


def _portfolio_summary(workloads: dict[str, Any]) -> dict[str, Any]:
    tensors = [tensor for workload in workloads.values() for tensor in workload["tensors"]]
    summary = _summarize_tensors(tensors)
    summary.pop("naive_assignment_space")
    summary.pop("naive_assignment_space_log10")
    largest_workload = max(
        workloads,
        key=lambda workload_id: int(workloads[workload_id]["opportunity"]["naive_assignment_space"]),
    )
    summary["workload_count"] = len(workloads)
    summary["workloads_with_nondegenerate_tensor"] = sum(
        workload["opportunity"]["nondegenerate_tensor_count"] > 0 for workload in workloads.values()
    )
    summary["largest_workload_assignment_space"] = {
        "workload": largest_workload,
        "size": workloads[largest_workload]["opportunity"]["naive_assignment_space"],
    }
    return summary


def _classify_opportunity(summary: dict[str, Any]) -> tuple[str, str, list[str]]:
    if summary["nondegenerate_tensor_count"] == 0:
        return "FAIL", "NO_NONDEGENERATE_TENSOR", ["NO_NONDEGENERATE_TENSOR"]
    limitations = []
    if summary["workloads_with_nondegenerate_tensor"] != summary["workload_count"]:
        limitations.append("WORKLOAD_WITHOUT_NONDEGENERATE_TENSOR")
    if summary["path_nondegenerate_count"] == 0:
        limitations.append("NO_INDEPENDENT_PATH_CHOICE")
    if limitations:
        return "NARROW", "PLACEMENT_ONLY", limitations
    return "PASS", "PLACEMENT_AND_PATH", limitations


def _file_digest(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


__all__ = ["OpportunityCensusError", "load_gate2b_contract", "run_opportunity_census"]
