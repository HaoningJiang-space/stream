"""Gate 2E-A potential-only census of finite operator execution templates."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
from itertools import product
from pathlib import Path
from typing import Any

from stream.structural.real_workload_lifting import _git_checked, _source_manifest, _source_run_manifest

_COMPUTATION_PREFIX = "ComputationNode:"
_DIMENSION_PATTERN = re.compile(r"d(\d+)")


class OperatorTemplateCensusError(RuntimeError):
    """The frozen potential-template census cannot be evaluated exactly."""


def load_operator_template_contract() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath("gate2e_operator_template_contract.json")
    contract = json.loads(resource.read_text(encoding="utf-8"))
    _validate_contract(contract)
    return contract


def run_operator_template_census(
    output_path: str | Path,
    *,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Count bounded syntactic templates without invoking STREAM stages or TTA."""

    contract = load_operator_template_contract()
    destination = Path(output_path).resolve()
    source_before = _source_manifest(source_commit, executed_module_path=__file__)
    source_gates = _load_source_gates(contract)
    workloads = {
        workload_id: _workload_census(workload_id, source_gates["gate2a_payload"]["workloads"][workload_id])
        for workload_id in contract["source_gates"]["required_workloads"]
    }
    source_after = _source_manifest(source_commit, executed_module_path=__file__)
    source = _source_run_manifest(source_before, source_after, destination)
    static_review = _static_review_manifest(contract, source)
    summary = _summary(workloads)
    observations = {
        "source_identified": bool(source["identified"]),
        "static_review_scope_match": bool(static_review["scope_match"]),
        "source_gate_hashes_match": True,
        "workload_denominator_match": set(workloads) == set(contract["source_gates"]["required_workloads"]),
        "baseline_compute_domains_singleton": all(
            operator["baseline_compute_domain_size"] == 1
            for workload in workloads.values()
            for group in workload["groups"]
            for operator in group["operators"]
        ),
        "candidate_templates_unique": all(
            operator["candidate_templates_unique"]
            for workload in workloads.values()
            for group in workload["groups"]
            for operator in group["operators"]
        ),
        "operator_nondegenerate_ratio": summary["operator_nondegenerate_ratio"],
        "workloads_with_potential_coupling_ratio": summary["workloads_with_potential_coupling_ratio"],
        "minimum_workload_max_group_log10_space": summary["minimum_workload_max_group_log10_space"],
    }
    expected = contract["pass_criteria"]
    if set(observations) != set(expected):
        raise OperatorTemplateCensusError("contract pass criteria are not in one-to-one correspondence with results")
    criteria = {
        name: observations[name] >= value if isinstance(value, float) else observations[name] == value
        for name, value in expected.items()
    }
    payload = {
        "contract": contract,
        "verdict": "DISCOVERY_PASS" if all(criteria.values()) else "DISCOVERY_FAIL",
        "evidence_class": "POTENTIAL_ONLY",
        "criteria": criteria,
        "observations": observations,
        "source_gates": {key: value for key, value in source_gates.items() if not key.endswith("_payload")},
        "source": source,
        "static_review": static_review,
        "workloads": workloads,
        "summary": summary,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def _workload_census(workload_id: str, workload_result: dict[str, Any]) -> dict[str, Any]:
    manifest = workload_result.get("manifest")
    if not workload_result.get("valid") or manifest is None:
        raise OperatorTemplateCensusError(f"{workload_id}: accepted Gate 2A lifting is unavailable")
    groups = [_group_census(workload_id, group) for group in manifest["groups"]]
    operators = [operator for group in groups for operator in group["operators"]]
    return {
        "family": workload_result["family"],
        "group_count": len(groups),
        "operator_count": len(operators),
        "nondegenerate_operator_count": sum(operator["candidate_state_count"] > 1 for operator in operators),
        "max_group_log10_space": max(group["naive_state_space_log10"] for group in groups),
        "potential_coupling_edge_count": sum(group["potential_coupling_edge_count"] for group in groups),
        "groups": groups,
    }


def _group_census(workload_id: str, group: dict[str, Any]) -> dict[str, Any]:
    mapping_nodes = group["mapping"]["nodes"]
    computation_nodes = [
        node for node in group["input_workload"]["nodes"] if node["id"].startswith(_COMPUTATION_PREFIX)
    ]
    operators = [_operator_census(workload_id, int(group["group"]), node, mapping_nodes) for node in computation_nodes]
    state_space = math.prod(operator["candidate_state_count"] for operator in operators)
    operator_ids = {operator["operator_id"] for operator in operators}
    coupling_edges = [
        edge
        for edge in group["input_workload"]["edges"]
        if edge["source"] in operator_ids and edge["target"] in operator_ids
    ]
    return {
        "group": int(group["group"]),
        "operator_count": len(operators),
        "nondegenerate_operator_count": sum(operator["candidate_state_count"] > 1 for operator in operators),
        "naive_state_space": str(state_space),
        "naive_state_space_log10": math.log10(state_space) if state_space else None,
        "potential_coupling_edge_count": len(coupling_edges),
        "potential_coupling_edges": coupling_edges,
        "operators": operators,
    }


def _operator_census(
    workload_id: str,
    group_id: int,
    node: dict[str, Any],
    mapping_nodes: dict[str, Any],
) -> dict[str, Any]:
    name = node["id"].removeprefix(_COMPUTATION_PREFIX)
    mapping = mapping_nodes.get(name)
    if mapping is None or mapping["node_type"] != "ComputationNode":
        raise OperatorTemplateCensusError(f"{workload_id}/group-{group_id}: missing computation mapping for {name}")
    baseline_options = mapping["resource_options"]
    if len(baseline_options) != 1:
        raise OperatorTemplateCensusError(
            f"{workload_id}/group-{group_id}/{name}: expected one baseline compute allocation"
        )
    core_pool = _parse_core_group(baseline_options[0])
    if not core_pool:
        raise OperatorTemplateCensusError(f"{workload_id}/group-{group_id}/{name}: empty baseline compute pool")
    output_count = len(node["outputs"])
    output_maps = node["operand_maps"][-output_count:] if output_count else []
    output_positions = sorted(
        {
            int(position)
            for affine_map in output_maps
            for position in _DIMENSION_PATTERN.findall(affine_map.split("->", maxsplit=1)[1])
        }
    )
    dimensions = node["dimensions"]
    if any(position >= len(dimensions) for position in output_positions):
        raise OperatorTemplateCensusError(f"{workload_id}/group-{group_id}/{name}: invalid output dimension")
    factors_by_position = [
        _divisors_up_to(int(dimensions[position]["size"]), len(core_pool)) for position in output_positions
    ]
    templates = []
    combinations = product(*factors_by_position) if factors_by_position else [()]
    for factors in combinations:
        core_count = math.prod(factors)
        if core_count > len(core_pool) or len(core_pool) % core_count:
            continue
        splits = [
            {"dimension": f"D{position}", "split": int(factor)}
            for position, factor in zip(output_positions, factors, strict=True)
            if factor > 1
        ]
        for offset in range(0, len(core_pool), core_count):
            templates.append(
                {
                    "splits": splits,
                    "core_group": list(core_pool[offset : offset + core_count]),
                    "core_count": core_count,
                }
            )
    keys = [_template_key(template) for template in templates]
    if not templates:
        raise OperatorTemplateCensusError(f"{workload_id}/group-{group_id}/{name}: no candidate templates")
    return {
        "operator_id": node["id"],
        "name": name,
        "operation": node["operation"],
        "baseline_compute_domain_size": len(baseline_options),
        "baseline_core_pool": list(core_pool),
        "baseline_tiling": mapping["tiling_options"],
        "output_indexed_dimensions": [
            {"dimension": f"D{position}", "extent": int(dimensions[position]["size"])}
            for position in output_positions
        ],
        "candidate_state_count": len(templates),
        "candidate_templates_unique": len(keys) == len(set(keys)),
        "candidate_templates_digest": _digest(keys),
        "candidate_templates": templates,
    }


def _summary(workloads: dict[str, Any]) -> dict[str, Any]:
    operators = [
        operator
        for workload in workloads.values()
        for group in workload["groups"]
        for operator in group["operators"]
    ]
    nondegenerate = sum(operator["candidate_state_count"] > 1 for operator in operators)
    workloads_with_coupling = sum(workload["potential_coupling_edge_count"] > 0 for workload in workloads.values())
    state_counts = [operator["candidate_state_count"] for operator in operators]
    return {
        "workload_count": len(workloads),
        "operator_count": len(operators),
        "nondegenerate_operator_count": nondegenerate,
        "operator_nondegenerate_ratio": nondegenerate / len(operators) if operators else 0.0,
        "workloads_with_potential_coupling": workloads_with_coupling,
        "workloads_with_potential_coupling_ratio": workloads_with_coupling / len(workloads) if workloads else 0.0,
        "potential_coupling_edge_count": sum(
            workload["potential_coupling_edge_count"] for workload in workloads.values()
        ),
        "minimum_workload_max_group_log10_space": min(
            (workload["max_group_log10_space"] for workload in workloads.values()), default=0.0
        ),
        "candidate_state_count_histogram": dict(sorted(Counter(map(str, state_counts)).items())),
        "candidate_state_count_min": min(state_counts, default=0),
        "candidate_state_count_max": max(state_counts, default=0),
        "candidate_state_count_sum": sum(state_counts),
    }


def _load_source_gates(contract: dict[str, Any]) -> dict[str, Any]:
    loaded = {}
    for gate_name in ("gate2a", "gate2d"):
        expected = contract["source_gates"][gate_name]
        path = Path(expected["artifact"])
        digest = _file_digest(path)
        if digest != expected["sha256"]:
            raise OperatorTemplateCensusError(f"{gate_name} artifact digest mismatch: {digest}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("verdict") != expected["required_verdict"]:
            raise OperatorTemplateCensusError(f"{gate_name} verdict mismatch")
        loaded[gate_name] = {"path": str(path), "sha256": digest, "verdict": payload["verdict"]}
        loaded[f"{gate_name}_payload"] = payload
    failed = sorted(name for name, passed in loaded["gate2d_payload"]["criteria"].items() if not passed)
    if failed != contract["source_gates"]["gate2d"]["required_failed_criteria"]:
        raise OperatorTemplateCensusError(f"Gate 2D failure boundary drifted: {failed}")
    workloads = contract["source_gates"]["required_workloads"]
    if set(loaded["gate2a_payload"]["workloads"]) != set(workloads):
        raise OperatorTemplateCensusError("Gate 2A workload denominator mismatch")
    return loaded


def _static_review_manifest(contract: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    expected = contract["static_review"]
    ok, changed = _git_checked("diff", "--name-only", f"{expected['base_commit']}..{source['commit']}")
    changed_paths = sorted(path for path in changed.splitlines() if path)
    expected_paths = expected["expected_changed_paths"]
    return {
        "base_commit": expected["base_commit"],
        "source_commit": source["commit"],
        "changed_paths": changed_paths,
        "expected_changed_paths": expected_paths,
        "scope_match": ok and changed_paths == expected_paths,
    }


def _validate_contract(contract: dict[str, Any]) -> None:
    expected_top_level = {
        "contract",
        "version",
        "design_status",
        "source_gates",
        "template_rules",
        "static_review",
        "pass_criteria",
        "excluded_claims",
    }
    if set(contract) != expected_top_level:
        raise OperatorTemplateCensusError("contract has missing or unknown top-level fields")
    expected_criteria = {
        "source_identified",
        "static_review_scope_match",
        "source_gate_hashes_match",
        "workload_denominator_match",
        "baseline_compute_domains_singleton",
        "candidate_templates_unique",
        "operator_nondegenerate_ratio",
        "workloads_with_potential_coupling_ratio",
        "minimum_workload_max_group_log10_space",
    }
    if set(contract["pass_criteria"]) != expected_criteria:
        raise OperatorTemplateCensusError("contract has missing, unknown, or unevaluated pass criteria")
    expected_paths = contract["static_review"]["expected_changed_paths"]
    if expected_paths != sorted(set(expected_paths)):
        raise OperatorTemplateCensusError("expected changed paths must be sorted and unique")


def _parse_core_group(value: str) -> tuple[int, ...]:
    if not value.startswith("cores:"):
        raise OperatorTemplateCensusError(f"invalid core-group key {value!r}")
    suffix = value.removeprefix("cores:")
    cores = tuple(sorted(int(core) for core in suffix.split(",") if core))
    if len(cores) != len(set(cores)):
        raise OperatorTemplateCensusError(f"duplicate core in {value!r}")
    return cores


def _divisors_up_to(size: int, limit: int) -> tuple[int, ...]:
    if size < 1 or limit < 1:
        raise OperatorTemplateCensusError("dimension extents and core limits must be positive")
    return tuple(factor for factor in range(1, min(size, limit) + 1) if size % factor == 0)


def _template_key(template: dict[str, Any]) -> str:
    return json.dumps(template, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return "sha256:" + sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


__all__ = ["OperatorTemplateCensusError", "load_operator_template_contract", "run_operator_template_census"]
