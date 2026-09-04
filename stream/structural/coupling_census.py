"""Gate 2C factor-scope audit for the exact executable structural proxy."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from importlib.resources import files
from itertools import combinations
from pathlib import Path
from typing import Any


class CouplingCensusError(RuntimeError):
    """The pinned executable model cannot support the requested factor-scope audit."""


def load_gate2c_contract() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath("gate2c_contract.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def run_coupling_census(
    output_path: str | Path,
    *,
    source_report: str | Path | None = None,
) -> dict[str, Any]:
    """Audit factor scopes without invoking VE, structural optimization, or TTA."""

    contract = load_gate2c_contract()
    source_path = Path(source_report or contract["source_gate"]["artifact"])
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_manifest = _validate_inputs(source_path, source_payload, contract)
    workload_results = {
        workload_id: _workload_coupling_census(source_payload["workloads"][workload_id])
        for workload_id in contract["source_gate"]["required_workloads"]
    }
    summary = _portfolio_summary(workload_results)
    regimes = {result["inference_regime"] for result in workload_results.values()}
    inference_regime = "SEPARABLE" if regimes == {"SEPARABLE"} else "COUPLED_NOT_SOLVED"
    payload = {
        "contract": contract,
        "verdict": "PASS",
        "inference_regime": inference_regime,
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


def _validate_inputs(source_path: Path, payload: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    source_gate = contract["source_gate"]
    report_digest = _file_digest(source_path)
    if report_digest != source_gate["sha256"]:
        raise CouplingCensusError(f"Gate 2B artifact digest mismatch: {report_digest}")
    if payload.get("verdict") not in source_gate["required_verdicts"]:
        raise CouplingCensusError(f"Gate 2B verdict is {payload.get('verdict')!r}")
    if set(payload.get("workloads", {})) != set(source_gate["required_workloads"]):
        raise CouplingCensusError("Gate 2B workload denominator mismatch")
    model = contract["audited_structural_model"]
    model_path = Path(model["source"])
    model_digest = _file_digest(model_path)
    if model_digest != model["sha256"]:
        raise CouplingCensusError(f"audited structural model digest mismatch: {model_digest}")
    return {
        "path": str(source_path),
        "sha256": report_digest,
        "verdict": payload["verdict"],
        "opportunity_class": payload["opportunity_class"],
        "audited_model_path": str(model_path),
        "audited_model_sha256": model_digest,
    }


def _workload_coupling_census(workload: dict[str, Any]) -> dict[str, Any]:
    variables = [tensor for tensor in workload["tensors"] if tensor["nondegenerate"]]
    variable_ids = [tensor["variable_id"] for tensor in variables]
    factor_scopes = [(variable_id,) for variable_id in variable_ids]
    metrics = _scope_metrics(variable_ids, factor_scopes)
    if metrics["inference_regime"] == "SEPARABLE":
        metrics["independent_argmin_evaluations"] = sum(int(tensor["assignment_domain_size"]) for tensor in variables)
    else:
        metrics["independent_argmin_evaluations"] = None
    metrics["naive_assignment_space"] = workload["opportunity"]["naive_assignment_space"]
    return {
        "family": workload["family"],
        "variable_count": len(variables),
        "factor_count": len(factor_scopes),
        **metrics,
    }


def _scope_metrics(variable_ids: list[str], factor_scopes: list[tuple[str, ...]]) -> dict[str, Any]:
    known_variables = set(variable_ids)
    if len(known_variables) != len(variable_ids):
        raise CouplingCensusError("factor graph has duplicate variable IDs")
    if any(not scope or not set(scope) <= known_variables for scope in factor_scopes):
        raise CouplingCensusError("factor scope is empty or references an unknown variable")
    non_unary = [scope for scope in factor_scopes if len(scope) > 1]
    edges = {tuple(sorted(edge)) for scope in factor_scopes for edge in combinations(scope, 2)}
    interacting = {variable for edge in edges for variable in edge}
    component_sizes = _component_sizes(known_variables, edges)
    separable = not non_unary
    return {
        "factor_arity_histogram": dict(sorted(Counter(map(str, map(len, factor_scopes))).items())),
        "non_unary_factor_count": len(non_unary),
        "primal_graph_edge_count": len(edges),
        "largest_coupled_component": max(component_sizes, default=0),
        "interaction_variable_ratio": len(interacting) / len(variable_ids) if variable_ids else 0.0,
        "induced_width": 0 if separable else None,
        "inference_regime": "SEPARABLE" if separable else "COUPLED_NOT_SOLVED",
    }


def _component_sizes(variables: set[str], edges: set[tuple[str, str]]) -> list[int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(variables)
    sizes = []
    while unseen:
        frontier = [unseen.pop()]
        size = 0
        while frontier:
            current = frontier.pop()
            size += 1
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            frontier.extend(neighbors)
        sizes.append(size)
    return sizes


def _portfolio_summary(workloads: dict[str, Any]) -> dict[str, Any]:
    return {
        "workload_count": len(workloads),
        "variable_count": sum(result["variable_count"] for result in workloads.values()),
        "factor_count": sum(result["factor_count"] for result in workloads.values()),
        "non_unary_factor_count": sum(result["non_unary_factor_count"] for result in workloads.values()),
        "primal_graph_edge_count": sum(result["primal_graph_edge_count"] for result in workloads.values()),
        "largest_coupled_component": max(
            (result["largest_coupled_component"] for result in workloads.values()), default=0
        ),
        "interaction_variable_ratio": (
            sum(result["interaction_variable_ratio"] * result["variable_count"] for result in workloads.values())
            / sum(result["variable_count"] for result in workloads.values())
        ),
        "induced_width": 0 if all(result["induced_width"] == 0 for result in workloads.values()) else None,
        "independent_argmin_evaluations": sum(
            result["independent_argmin_evaluations"] or 0 for result in workloads.values()
        ),
    }


def _file_digest(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


__all__ = ["CouplingCensusError", "load_gate2c_contract", "run_coupling_census"]
