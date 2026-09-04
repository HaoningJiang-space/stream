"""Gate 2D deterministic route-option expansion census."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from stream.execution_boundary import ExecutionEvent
from stream.structural.opportunity_census import _workload_census
from stream.structural.real_workload_lifting import (
    _environment_manifest,
    _git_checked,
    _run_preparation_attempts,
    _source_manifest,
    _source_run_manifest,
    load_gate2a_contract,
)

_SUPPORTED_PASS_CRITERIA = frozenset(
    {
        "source_identified",
        "environment_compatible",
        "workload_denominator_match",
        "static_review_scope_match",
        "baseline_semantic_hash_match",
        "structural_invariant_hash_match",
        "deterministic_preparation",
        "lifting_success",
        "forbidden_execution_events",
        "same_tensor_and_transfer_variables",
        "placement_domains_identical",
        "route_resource_signatures_unique",
        "path_domains_included",
        "compatible_path_counts_monotone",
        "strict_path_domain_expansion",
        "expanded_path_nondegenerate_tensor_ratio",
        "expanded_controlled_logical_bits_ratio",
    }
)
_MIN_REPEAT_COUNT = 2


class RouteOptionGateError(RuntimeError):
    """The pinned route-expansion experiment cannot be evaluated exactly."""


def load_gate2d_contract() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath("gate2d_contract.json")
    contract = json.loads(resource.read_text(encoding="utf-8"))
    _validate_contract(contract)
    return contract


def run_route_option_gate(
    output_path: str | Path,
    *,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Compare baseline and expanded route domains without constructing or solving TTA."""

    contract = load_gate2d_contract()
    gate2a_contract = load_gate2a_contract()
    denominator = _workload_denominator_manifest(contract, gate2a_contract)
    source_gate = _load_source_gate(contract)
    destination = Path(output_path).resolve()
    source_before = _source_manifest(source_commit, executed_module_path=__file__)
    environment = _environment_manifest(gate2a_contract)
    hardware_path = Path(gate2a_contract["hardware"])
    repeat_count = int(contract["repeat_count"])
    max_workers = int(contract["max_parallel_workers"])
    results = {}
    run_started = perf_counter()
    with TemporaryDirectory(prefix="stream-gate2d-") as temporary:
        temporary_root = Path(temporary)
        for plan_limit in contract["route_plan_limits"]:
            raw_results = _run_preparation_attempts(
                gate2a_contract["workloads"],
                hardware_path,
                temporary_root / f"plans-{plan_limit}",
                repeat_count,
                max_workers,
                int(plan_limit),
            )
            results[str(plan_limit)] = {
                workload_id: _compact_workload_result(workload_id, result)
                for workload_id, result in raw_results.items()
            }
    wall_seconds = perf_counter() - run_started
    source_after = _source_manifest(source_commit, executed_module_path=__file__)
    source = _source_run_manifest(source_before, source_after, destination)
    static_review = _static_review_manifest(contract, source)
    criteria, observations = _criteria(
        contract,
        source_gate,
        source,
        environment,
        denominator,
        static_review,
        results,
    )
    payload = {
        "contract": contract,
        "verdict": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria,
        "source_gate": source_gate,
        "source": source,
        "environment": environment,
        "workload_denominator": denominator,
        "static_review": static_review,
        "observations": observations,
        "execution": {
            "attempt_count": len(gate2a_contract["workloads"]) * repeat_count * len(contract["route_plan_limits"]),
            "max_workers_per_limit": max_workers,
            "wall_seconds": round(wall_seconds, 6),
        },
        "results": results,
        "comparison": _comparison(contract, results),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


def _load_source_gate(contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["source_gate"]
    path = Path(expected["artifact"])
    digest = _file_digest(path)
    if digest != expected["sha256"]:
        raise RouteOptionGateError(f"Gate 2A artifact digest mismatch: {digest}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("verdict") != expected["required_verdict"] or not all(payload.get("criteria", {}).values()):
        raise RouteOptionGateError("Gate 2A source evidence is not passing")
    if set(payload.get("workloads", {})) != set(expected["required_workloads"]):
        raise RouteOptionGateError("Gate 2A workload denominator mismatch")
    return {
        "path": str(path),
        "sha256": digest,
        "workload_semantic_hashes": {
            workload_id: payload["workloads"][workload_id]["semantic_hash"]
            for workload_id in expected["required_workloads"]
        },
    }


def _compact_workload_result(workload_id: str, result: dict[str, Any]) -> dict[str, Any]:
    manifest = result["manifest"]
    if manifest is None:
        return {
            "family": result["family"],
            "valid": False,
            "deterministic": result["deterministic"],
            "semantic_hash": result["semantic_hash"],
            "attempts": result["attempts"],
            "execution_boundary": None,
            "opportunity": None,
            "tensors": [],
            "domain_evidence": None,
            "structural_invariant_hash": None,
        }
    census = _workload_census(workload_id, result)
    return {
        "family": result["family"],
        "valid": result["valid"],
        "deterministic": result["deterministic"],
        "semantic_hash": result["semantic_hash"],
        "attempts": result["attempts"],
        "execution_boundary": manifest["execution_boundary"],
        "opportunity": census["opportunity"],
        "tensors": census["tensors"],
        "domain_evidence": _domain_evidence(manifest),
        "structural_invariant_hash": _structural_invariant_hash(manifest),
    }


def _domain_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project canonical route identities without changing the accepted Gate 2A semantic hash."""

    transfers = {}
    staging_tensors = {}
    for group in manifest["groups"]:
        group_id = int(group["group"])
        domains = group["domains"]
        tensor_rows = {row["id"]: row for row in domains["tensors"]}
        for transfer in domains["transfers"]:
            variable_id = f"group-{group_id}:{transfer['id']}"
            path_keys = tuple(transfer["paths"])
            transfers[variable_id] = {
                "path_keys": list(path_keys),
                "route_resource_signatures": [_route_resource_signature(path) for path in path_keys],
            }
        for staging in domains["staging_tensors"]:
            tensor = tensor_rows[staging["tensor"]]
            adjacent = tuple(f"group-{group_id}:{transfer}" for transfer in tensor["adjacent_transfers"])
            choices = {}
            for choice in staging["choices"]:
                counts = tuple(int(count) for count in choice["compatible_path_counts"])
                if len(counts) != len(adjacent):
                    raise RouteOptionGateError(
                        f"group-{group_id}:{staging['tensor']} has misaligned transfer/path-count evidence"
                    )
                placement = choice["placement"]
                if placement in choices:
                    raise RouteOptionGateError(
                        f"group-{group_id}:{staging['tensor']} has duplicate placement {placement!r}"
                    )
                choices[placement] = {
                    "assignment_count": int(choice["assignment_count"]),
                    "compatible_path_counts": dict(zip(adjacent, counts, strict=True)),
                }
            staging_tensors[f"group-{group_id}:{staging['tensor']}"] = {
                "adjacent_transfers": list(adjacent),
                "placements": sorted(choices),
                "choices": choices,
                "assignment_count": int(staging["assignment_count"]),
            }
    return {
        "transfers": dict(sorted(transfers.items())),
        "staging_tensors": dict(sorted(staging_tensors.items())),
    }


def _structural_invariant_hash(manifest: dict[str, Any]) -> str:
    groups = []
    for group in manifest["groups"]:
        mapping = group["mapping"]
        mapping_nodes = {}
        for node_id, node in mapping["nodes"].items():
            mapping_nodes[node_id] = (
                node
                if node["node_type"] != "TransferNode"
                else {key: value for key, value in node.items() if key != "resource_options"}
            )
        groups.append(
            {
                "group": group["group"],
                "group_trace": group["group_trace"],
                "preparation_trace": group["preparation_trace"],
                "input_workload": group["input_workload"],
                "steady_state_workload": group["steady_state_workload"],
                "mapping": {
                    "fused_groups": mapping["fused_groups"],
                    "nodes": mapping_nodes,
                    "runtime_args": mapping["runtime_args"],
                },
                "iterations": group["iterations"],
                "multiplicities": group["multiplicities"],
                "ssis": group["ssis"],
                "tensor_domains": group["domains"]["tensors"],
                "transfer_ids": sorted(transfer["id"] for transfer in group["domains"]["transfers"]),
                "mapping_fallbacks": group["mapping_fallbacks"],
            }
        )
    projection = {
        "frontend_trace": manifest["frontend_trace"],
        "semantic_exclusions_audited": manifest["semantic_exclusions_audited"],
        "semantic_exclusions": manifest["semantic_exclusions"],
        "group_count": manifest["group_count"],
        "groups": groups,
    }
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + sha256(encoded).hexdigest()


def _route_resource_signature(path_key: str) -> str:
    path = json.loads(path_key)
    if set(path) != {"sources", "targets", "total_hops_objective", "links"}:
        raise RouteOptionGateError("unexpected canonical route-plan schema")
    resource_projection = {
        "sources": sorted(path["sources"]),
        "targets": sorted(path["targets"]),
        "links": sorted(path["links"], key=lambda link: json.dumps(link, sort_keys=True)),
    }
    return json.dumps(resource_projection, sort_keys=True, separators=(",", ":"))


def _comparison(contract: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    baseline_key = str(contract["expansion"]["baseline_limit"])
    expanded_key = str(contract["expansion"]["expanded_limit"])
    baseline = results[baseline_key]
    expanded = results[expanded_key]
    rows = {}
    for workload_id in contract["source_gate"]["required_workloads"]:
        baseline_evidence = baseline[workload_id]["domain_evidence"] or {"transfers": {}, "staging_tensors": {}}
        expanded_evidence = expanded[workload_id]["domain_evidence"] or {"transfers": {}, "staging_tensors": {}}
        baseline_tensors = baseline_evidence["staging_tensors"]
        expanded_tensors = expanded_evidence["staging_tensors"]
        baseline_transfers = baseline_evidence["transfers"]
        expanded_transfers = expanded_evidence["transfers"]
        same_tensor_variables = set(baseline_tensors) == set(expanded_tensors)
        same_transfer_variables = set(baseline_transfers) == set(expanded_transfers)
        placement_domains_identical = same_tensor_variables and all(
            baseline_tensors[variable]["placements"] == expanded_tensors[variable]["placements"]
            for variable in baseline_tensors
        )
        route_resource_signatures_unique = same_transfer_variables and all(
            len(row["route_resource_signatures"]) == len(set(row["route_resource_signatures"]))
            for domains in (baseline_transfers, expanded_transfers)
            for row in domains.values()
        )
        path_domains_included = same_transfer_variables and all(
            set(baseline_transfers[transfer]["route_resource_signatures"])
            <= set(expanded_transfers[transfer]["route_resource_signatures"])
            for transfer in baseline_transfers
        )
        compatible_path_counts_monotone = (
            same_tensor_variables
            and placement_domains_identical
            and all(
                _compatible_path_counts_included(baseline_tensors[variable], expanded_tensors[variable])
                for variable in baseline_tensors
            )
        )
        strict_path_domain_expansion = same_tensor_variables and all(
            baseline_tensors[variable]["assignment_count"] < expanded_tensors[variable]["assignment_count"]
            for variable in baseline_tensors
        )
        opportunity = expanded[workload_id]["opportunity"]
        eligible_tensors = opportunity["eligible_tensor_count"] if opportunity else 0
        rows[workload_id] = {
            "same_tensor_variables": same_tensor_variables,
            "same_transfer_variables": same_transfer_variables,
            "placement_domains_identical": placement_domains_identical,
            "route_resource_signatures_unique": route_resource_signatures_unique,
            "path_domains_included": path_domains_included,
            "compatible_path_counts_monotone": compatible_path_counts_monotone,
            "strict_path_domain_expansion": strict_path_domain_expansion,
            "structural_invariant_hash_match": (
                baseline[workload_id]["structural_invariant_hash"]
                == expanded[workload_id]["structural_invariant_hash"]
            ),
            "expanded_path_nondegenerate_tensor_ratio": (
                opportunity["path_nondegenerate_count"] / eligible_tensors if eligible_tensors else 0.0
            ),
            "expanded_controlled_logical_bits_ratio": (
                opportunity["controlled_logical_bits_ratio"] if opportunity else 0.0
            ),
        }
    return rows


def _compatible_path_counts_included(baseline: dict[str, Any], expanded: dict[str, Any]) -> bool:
    for placement in baseline["placements"]:
        baseline_counts = baseline["choices"][placement]["compatible_path_counts"]
        expanded_counts = expanded["choices"][placement]["compatible_path_counts"]
        if set(baseline_counts) != set(expanded_counts):
            return False
        if any(int(baseline_counts[transfer]) > int(expanded_counts[transfer]) for transfer in baseline_counts):
            return False
    return True


def _criteria(
    contract: dict[str, Any],
    source_gate: dict[str, Any],
    source: dict[str, Any],
    environment: dict[str, Any],
    denominator: dict[str, Any],
    static_review: dict[str, Any],
    results: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    baseline_key = str(contract["expansion"]["baseline_limit"])
    all_results = [result for by_workload in results.values() for result in by_workload.values()]
    comparison = _comparison(contract, results)
    execution_boundaries = [result["execution_boundary"] for result in all_results]
    boundary_schema_valid = all(
        boundary is not None and set(boundary) == {event.value for event in ExecutionEvent}
        for boundary in execution_boundaries
    )
    forbidden_execution_events = (
        sum(sum(boundary.values()) for boundary in execution_boundaries if boundary is not None)
        if boundary_schema_valid
        else None
    )
    baseline_hashes = source_gate["workload_semantic_hashes"]
    observations = {
        "source_identified": bool(source["identified"]),
        "environment_compatible": bool(environment["compatible"]),
        "workload_denominator_match": bool(denominator["matches_frozen_denominator"]),
        "static_review_scope_match": bool(static_review["scope_match"]),
        "baseline_semantic_hash_match": all(
            results[baseline_key][workload_id]["semantic_hash"] == semantic_hash
            for workload_id, semantic_hash in baseline_hashes.items()
        ),
        "structural_invariant_hash_match": all(
            row["structural_invariant_hash_match"] for row in comparison.values()
        ),
        "deterministic_preparation": all(result["deterministic"] for result in all_results),
        "lifting_success": all(result["valid"] for result in all_results),
        "forbidden_execution_events": forbidden_execution_events,
        "same_tensor_and_transfer_variables": all(
            row["same_tensor_variables"] and row["same_transfer_variables"] for row in comparison.values()
        ),
        "placement_domains_identical": all(row["placement_domains_identical"] for row in comparison.values()),
        "route_resource_signatures_unique": all(
            row["route_resource_signatures_unique"] for row in comparison.values()
        ),
        "path_domains_included": all(row["path_domains_included"] for row in comparison.values()),
        "compatible_path_counts_monotone": all(
            row["compatible_path_counts_monotone"] for row in comparison.values()
        ),
        "strict_path_domain_expansion": all(
            row["strict_path_domain_expansion"] for row in comparison.values()
        ),
        "expanded_path_nondegenerate_tensor_ratio": min(
            (row["expanded_path_nondegenerate_tensor_ratio"] for row in comparison.values()), default=0.0
        ),
        "expanded_controlled_logical_bits_ratio": min(
            (row["expanded_controlled_logical_bits_ratio"] for row in comparison.values()), default=0.0
        ),
    }
    expected = contract["pass_criteria"]
    if set(observations) != set(expected) or set(expected) != _SUPPORTED_PASS_CRITERIA:
        raise RouteOptionGateError("Gate 2D criteria are not in one-to-one correspondence with the contract")
    criteria = {name: observations[name] == expected[name] for name in expected}
    return criteria, observations


def _workload_denominator_manifest(contract: dict[str, Any], gate2a_contract: dict[str, Any]) -> dict[str, Any]:
    frozen = tuple(contract["source_gate"]["required_workloads"])
    current = tuple(workload["id"] for workload in gate2a_contract["workloads"])
    matches = len(current) == len(set(current)) and current == frozen
    if not matches:
        raise RouteOptionGateError(f"Gate 2A workload denominator drifted: frozen={frozen}, current={current}")
    return {
        "frozen": list(frozen),
        "current": list(current),
        "matches_frozen_denominator": True,
    }


def _static_review_manifest(contract: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    static_contract = contract["static_review"]
    ok, changed = _git_checked("diff", "--name-only", f"{static_contract['base_commit']}..{source['commit']}")
    changed_paths = sorted(path for path in changed.splitlines() if path)
    expected_paths = sorted(static_contract["expected_changed_paths"])
    return {
        "base_commit": static_contract["base_commit"],
        "source_commit": source["commit"],
        "changed_paths": changed_paths,
        "expected_changed_paths": expected_paths,
        "scope_match": ok and changed_paths == expected_paths,
        "review_only_invariants": static_contract["review_only_invariants"],
        "evidence_boundary": static_contract["evidence_boundary"],
    }


def _validate_contract(contract: dict[str, Any]) -> None:
    required_top_level = {
        "contract",
        "version",
        "design_status",
        "source_gate",
        "route_plan_limits",
        "repeat_count",
        "max_parallel_workers",
        "expansion",
        "static_review",
        "pass_criteria",
        "excluded_claims",
    }
    if set(contract) != required_top_level:
        raise RouteOptionGateError("Gate 2D contract has missing or unknown top-level fields")
    expected_nested_fields = {
        "source_gate": {"artifact", "sha256", "required_verdict", "required_workloads"},
        "expansion": {"scope", "baseline_limit", "expanded_limit", "measured_invariants"},
        "static_review": {
            "base_commit",
            "expected_changed_paths",
            "review_only_invariants",
            "evidence_boundary",
        },
    }
    if any(set(contract[section]) != fields for section, fields in expected_nested_fields.items()):
        raise RouteOptionGateError("Gate 2D contract has missing or unknown nested fields")
    if set(contract["pass_criteria"]) != _SUPPORTED_PASS_CRITERIA:
        raise RouteOptionGateError("Gate 2D contract has missing, unknown, or unevaluated pass criteria")
    limits = contract["route_plan_limits"]
    if (
        limits != [contract["expansion"]["baseline_limit"], contract["expansion"]["expanded_limit"]]
        or any(type(limit) is not int or limit < 1 for limit in limits)
        or limits[0] >= limits[1]
    ):
        raise RouteOptionGateError("Gate 2D route-plan limits must be ordered positive integers")
    if type(contract["repeat_count"]) is not int or contract["repeat_count"] < _MIN_REPEAT_COUNT:
        raise RouteOptionGateError("Gate 2D repeat_count must be an integer of at least two")
    if type(contract["max_parallel_workers"]) is not int or contract["max_parallel_workers"] < 1:
        raise RouteOptionGateError("Gate 2D max_parallel_workers must be a positive integer")
    frozen_workloads = contract["source_gate"]["required_workloads"]
    if not frozen_workloads or len(frozen_workloads) != len(set(frozen_workloads)):
        raise RouteOptionGateError("Gate 2D frozen workload IDs must be non-empty and unique")
    expected_paths = contract["static_review"]["expected_changed_paths"]
    if not expected_paths or expected_paths != sorted(set(expected_paths)):
        raise RouteOptionGateError("Gate 2D expected changed paths must be sorted, non-empty, and unique")


def _file_digest(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


__all__ = ["RouteOptionGateError", "load_gate2d_contract", "run_route_option_gate"]
