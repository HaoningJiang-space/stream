"""Gate 1A-v3: paired operator-template compiler faithfulness."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distributions, version
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from stream.cost_model.core_cost import CoreCostEntry
from stream.cost_model.core_cost_lut import CoreCostLUT
from stream.cost_model.steady_state_scheduler import SteadyStateScheduler
from stream.datatypes import LayerDim
from stream.mapping.mapping import FusedGroup, Mapping, NodeMapping
from stream.opt.solver import ConstraintSelection
from stream.stages.context import StageContext
from stream.stages.generation.kernel_state import KernelStateStage
from stream.stages.generation.operator_template_compilation import OperatorTemplateCompilationStage
from stream.stages.generation.tiling_generation import TilingGenerationStage
from stream.stages.stage import LeafStage, MainStage
from stream.structural.gate1a_cases import build_gate1a_workload, load_gate1a_accelerator
from stream.structural.operator_template_contract import (
    OperatorTemplate,
    OperatorTemplateAssignment,
    OperatorTemplateLibrary,
)
from stream.structural.pipeline import prepare_uninstrumented_reference_problem
from stream.structural.stream_contract import Gate1AEvalConfig, canonical_mapping_manifest, tiling_key

_CONTRACT_VERSION = 3
_MIN_COMPILE_REPETITIONS = 2


class OperatorTemplateFaithfulnessError(RuntimeError):
    """The frozen Gate 1A-v3 contract or evidence is incomplete."""


def load_operator_template_faithfulness_contract() -> dict[str, Any]:
    resource = files("stream.structural.contracts").joinpath("gate1a_operator_template_v3.json")
    contract = json.loads(resource.read_text(encoding="utf-8"))
    _validate_contract(contract)
    return contract


def run_operator_template_faithfulness(
    output_path: str | Path,
    *,
    source_commit: str | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Compare every compiled micro assignment with an independent singleton reference."""

    contract = load_operator_template_faithfulness_contract()
    source_gate = _load_source_gate(contract)
    destination = Path(output_path).resolve()
    source_before = _source_manifest(source_commit)
    started = perf_counter()
    specs = _assignment_specs(contract)
    required_workers = contract["max_workers"]
    if max_workers is not None and max_workers != required_workers:
        raise ValueError(f"Gate 1A-v3 requires exactly {required_workers} parallel workers")
    worker_count = min(required_workers, len(specs))
    accelerator = load_gate1a_accelerator(contract["hardware"])
    with TemporaryDirectory(prefix="stream-gate1a-v3-") as temporary:
        root = Path(temporary)
        results = _run_assignments(specs, root / "parallel", accelerator, contract, worker_count)
        serial_results = _run_assignments(specs, root / "serial", accelerator, contract, 1)
    replay_exact = results == serial_results
    wall_seconds = perf_counter() - started
    source_after = _source_manifest(source_commit)
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
        raise OperatorTemplateFaithfulnessError("contract criteria do not match computed observations")
    criteria = {name: observations[name] == value for name, value in expected.items()}
    exact_coverage = _summary(results)["exact_executable_coverage"]
    if not all(criteria.values()):
        verdict = "FAIL"
    elif exact_coverage == 1.0:
        verdict = "PASS"
    else:
        verdict = "NARROW"
    payload = {
        "contract": contract,
        "verdict": verdict,
        "evidence_class": "MICRO_EXACT_PRE_TILING_COMPILER",
        "criteria": criteria,
        "observations": observations,
        "source_gate": source_gate,
        "source": source,
        "environment": _environment_manifest(),
        "static_review": static_review,
        "execution": {
            "assignment_count": len(specs),
            "worker_count": worker_count,
            "serial_replay_worker_count": 1,
            "wall_seconds": round(wall_seconds, 6),
            "tta_solve_invoked": False,
            "serial_parallel_replay_exact": replay_exact,
        },
        "summary": _summary(results),
        "assignments": results,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(destination)
    return payload


def _assignment_specs(contract: dict[str, Any]) -> list[tuple[str, str, int]]:
    specs = []
    for dag_class, expected_operators in contract["dag_classes"].items():
        workload = build_gate1a_workload(dag_class)
        names = sorted(node.name for node in workload.get_computation_nodes())
        if len(names) != expected_operators:
            raise OperatorTemplateFaithfulnessError(
                f"{dag_class}: expected {expected_operators} operators, observed {len(names)}"
            )
        specs.extend((dag_class, name, index) for name in names for index in range(len(contract["template_family"])))
    if len(specs) != contract["expected_assignment_count"]:
        raise OperatorTemplateFaithfulnessError("assignment denominator does not match the frozen contract")
    return specs


def _run_assignments(specs, temporary_root, accelerator, contract, worker_count) -> list[dict[str, Any]]:
    results: dict[tuple[str, str, int], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="gate1a-v3") as executor:
        pending = {
            executor.submit(
                _run_assignment,
                dag_class,
                target,
                template_index,
                temporary_root / f"{dag_class}-{target}-{template_index}",
                accelerator,
                contract,
            ): (dag_class, target, template_index)
            for dag_class, target, template_index in specs
        }
        for future in as_completed(pending):
            results[pending[future]] = future.result()
    return [results[spec] for spec in specs]


def _run_assignment(dag_class, target, template_index, output_root, accelerator, contract) -> dict[str, Any]:
    workload = build_gate1a_workload(dag_class)
    baseline = _baseline_mapping(workload, accelerator)
    library = _template_library(workload, contract)
    template = next(
        item
        for item in library.templates
        if item.target == target and item == _template_from_spec(target, contract["template_family"][template_index])
    )
    assignment_id = f"{dag_class}:{target}:{template_index}"
    assignment = OperatorTemplateAssignment(assignment_id, (template,))
    compiled_context = _run_tiling_pipeline(
        workload,
        baseline,
        output_root / "compiled",
        assignment=assignment,
        library=library,
    )
    reference_mapping = _direct_reference_mapping(workload, baseline, template)
    reference_mapping_is_baseline = canonical_mapping_manifest(reference_mapping) == canonical_mapping_manifest(
        baseline
    )
    reference_context = _run_tiling_pipeline(workload, reference_mapping, output_root / "reference")

    compilation = compiled_context.get("operator_template_compilation")
    if compilation is None:
        raise OperatorTemplateFaithfulnessError(f"{assignment_id}: compiler stage did not record a result")
    compiled_pair = _selected_pair(compiled_context.get("mapping"), target)
    reference_pair = _selected_pair(reference_context.get("mapping"), target)
    expected_pair = _expected_pair(template)
    compiled_outcome = _prepared_problem_outcome(compiled_context, accelerator, output_root / "compiled-scheduler")
    reference_outcome = _prepared_problem_outcome(reference_context, accelerator, output_root / "reference-scheduler")
    hashes = []
    for _ in range(contract["compile_repetitions"]):
        repeated = _run_compiler_only(workload, baseline, assignment, library)
        hashes.append(repeated.semantic_hash)
    exact_pair = compiled_pair == reference_pair == expected_pair
    outcome_exact = compiled_outcome == reference_outcome
    baseline_template = _template_from_spec(target, contract["template_family"][-1])
    baseline_round_trip = None
    if template == baseline_template:
        baseline_round_trip = (
            reference_mapping_is_baseline
            and outcome_exact
            and compiled_outcome["status"] == "EXACT_EXECUTABLE"
        )
    silent_relaxations = int(not exact_pair) + int(not outcome_exact)
    return {
        "assignment_id": assignment_id,
        "dag_class": dag_class,
        "target": target,
        "template": template.key,
        "paired_candidate_set_exact": exact_pair,
        "compiled_pair": compiled_pair,
        "reference_pair": reference_pair,
        "expected_pair": expected_pair,
        "compiled_reference_outcome_exact": outcome_exact,
        "compiled_outcome": compiled_outcome,
        "reference_outcome": reference_outcome,
        "exact_executable": compiled_outcome["status"] == "EXACT_EXECUTABLE",
        "compiler_hashes": hashes,
        "compiler_deterministic": len(set(hashes)) == 1,
        "baseline_round_trip_exact": baseline_round_trip,
        "silent_relaxations": silent_relaxations,
    }


def _run_tiling_pipeline(workload, mapping, output_path, *, assignment=None, library=None) -> StageContext:
    output_path.mkdir(parents=True, exist_ok=True)
    context = StageContext.from_kwargs(
        workload=workload,
        mapping=mapping,
        output_path=str(output_path),
        operator_template_assignment=assignment,
        operator_template_library=library,
    )
    stages = [OperatorTemplateCompilationStage, KernelStateStage, TilingGenerationStage, LeafStage]
    if assignment is None:
        stages = [KernelStateStage, TilingGenerationStage, LeafStage]
    results = MainStage(stages, context).run()
    if len(results) != 1:
        raise OperatorTemplateFaithfulnessError(f"tiling pipeline returned {len(results)} contexts")
    return results[0]


def _run_compiler_only(workload, mapping, assignment, library):
    context = StageContext.from_kwargs(
        workload=workload,
        mapping=mapping,
        operator_template_assignment=assignment,
        operator_template_library=library,
    )
    results = MainStage([OperatorTemplateCompilationStage, LeafStage], context).run()
    return results[0].require_value("operator_template_compilation", "Gate1A-v3")


def _prepared_problem_outcome(context: StageContext, accelerator, output_path: Path) -> dict[str, Any]:
    workload = context.require_value("workload", "Gate1A-v3")
    mapping = context.require_value("mapping", "Gate1A-v3")
    cost_lut = CoreCostLUT(load=False)
    for node in workload.get_computation_nodes():
        allocation = mapping.get(node).resource_allocation
        if len(allocation) != 1 or not allocation[0]:
            raise OperatorTemplateFaithfulnessError(f"{node.name}: expected one non-empty compiled allocation")
        for core in allocation[0]:
            cost_lut.add_cost(node, core, CoreCostEntry(1, 1, 1, 1))
    constraints = ConstraintSelection()
    scheduler = SteadyStateScheduler(
        workload,
        accelerator,
        mapping,
        context.require_value("fusion_splits", "Gate1A-v3"),
        cost_lut,
        output_path=str(output_path),
        backend="ORTOOLS_GSCIP",
        constraint_selection=constraints,
        max_transfer_plans_per_endpoint=1,
    )
    eval_config = Gate1AEvalConfig(
        backend="ORTOOLS_GSCIP",
        constraints=("memory_capacity", "object_fifo_depth", "buffer_descriptors", "dma_channels", "pipelining"),
    )
    try:
        problem_hash = prepare_uninstrumented_reference_problem(scheduler, eval_config).compilation.problem_hash
    except ValueError as error:
        detail = str(error)
        if "Shared-input demand has incompatible tensor-relevant tilings" in detail:
            return {
                "status": "UNSUPPORTED",
                "reason": "SHARED_INPUT_TILING_INCOMPATIBLE",
                "failure_signature": detail,
            }
        return {
            "status": "PREPARATION_FAILURE",
            "reason": type(error).__name__,
            "detail": detail,
        }
    return {"status": "EXACT_EXECUTABLE", "problem_hash": problem_hash}


def _baseline_mapping(workload, accelerator) -> Mapping:
    cores = tuple(accelerator.get_core(index) for index in range(4))
    if any(core is None for core in cores):
        raise OperatorTemplateFaithfulnessError("frozen micro accelerator lacks cores 0..3")
    local_dim = LayerDim(position=0, prefix="d")
    nodes = workload.get_computation_nodes()
    first_dim = workload.get_dims(nodes[0])[0]
    tile, remainder = divmod(workload.get_dimension_size(first_dim), len(cores))
    if remainder:
        raise OperatorTemplateFaithfulnessError("micro dimension is not divisible by the baseline core count")
    mapping = Mapping(
        fused_groups=(FusedGroup("Gate1A-v3", tuple(node.name for node in nodes), ((first_dim, tile),)),)
    )
    for node in nodes:
        mapping.set(
            node,
            NodeMapping(
                resource_allocation=(cores,),
                inter_core_tiling=(((local_dim, len(cores)),),),
            ),
        )
    return mapping


def _template_library(workload, contract) -> OperatorTemplateLibrary:
    return OperatorTemplateLibrary(
        tuple(
            _template_from_spec(node.name, spec)
            for node in workload.get_computation_nodes()
            for spec in contract["template_family"]
        )
    )


def _template_from_spec(target: str, spec: dict[str, Any]) -> OperatorTemplate:
    return OperatorTemplate(
        target,
        tuple(int(core) for core in spec["core_ids"]),
        tuple((int(position), int(factor)) for position, factor in spec["splits"]),
    )


def _direct_reference_mapping(workload, baseline: Mapping, template: OperatorTemplate) -> Mapping:
    """Independent singleton construction; deliberately does not call the compiler."""

    return _direct_reference_mapping_for_templates(workload, baseline, (template,))


def _direct_reference_mapping_for_templates(
    workload,
    baseline: Mapping,
    templates: tuple[OperatorTemplate, ...],
) -> Mapping:
    """Independently construct a mapping for one or more atomic template choices."""

    reference = Mapping(fused_groups=baseline.fused_groups, runtime_args=dict(baseline.runtime_args))
    template_by_target = {template.target: template for template in templates}
    if len(template_by_target) != len(templates):
        raise OperatorTemplateFaithfulnessError("direct reference templates must have unique targets")
    for node in workload.get_computation_nodes():
        source = baseline.get(node)
        template = template_by_target.get(node.name)
        if template is None:
            reference.set(
                node,
                NodeMapping(
                    resource_allocation=tuple(source.resource_allocation),
                    inter_core_tiling=tuple(source.inter_core_tiling),
                    memory_allocation=tuple(source.memory_allocation),
                    kernel=source.kernel,
                ),
            )
            continue
        core_by_id = {core.id: core for core in source.resource_allocation[0]}
        tiling = tuple((LayerDim(position=position, prefix="d"), factor) for position, factor in template.splits)
        reference.set(
            node,
            NodeMapping(
                resource_allocation=(tuple(core_by_id[core_id] for core_id in template.core_ids),),
                inter_core_tiling=(tiling,) if tiling else (),
                memory_allocation=tuple(source.memory_allocation),
                kernel=source.kernel,
            ),
        )
    return reference


def _selected_pair(mapping: Mapping, target: str) -> dict[str, Any]:
    matches = [node for node in mapping.nodes() if node.name == target]
    if len(matches) != 1:
        raise OperatorTemplateFaithfulnessError(f"{target}: compiled mapping target is not unique")
    node = matches[0]
    node_mapping = mapping.get(node)
    return {
        "cores": [core.id for core in node_mapping.resource_allocation[0]],
        "tiling": [tiling_key(option) for option in node_mapping.inter_core_tiling],
    }


def _expected_pair(template: OperatorTemplate) -> dict[str, Any]:
    tiling = tuple((LayerDim(position=position, prefix="d"), factor) for position, factor in template.splits)
    return {"cores": list(template.core_ids), "tiling": [tiling_key(tiling)] if tiling else []}


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    unsupported_reasons: dict[str, int] = {}
    for result in results:
        outcome = result["compiled_outcome"]
        if outcome["status"] == "UNSUPPORTED":
            reason = outcome["reason"]
            unsupported_reasons[reason] = unsupported_reasons.get(reason, 0) + 1
    return {
        "assignment_count": len(results),
        "paired_candidate_set_exact_count": sum(result["paired_candidate_set_exact"] for result in results),
        "compiled_reference_outcome_exact_count": sum(
            result["compiled_reference_outcome_exact"] for result in results
        ),
        "exact_executable_count": sum(result["exact_executable"] for result in results),
        "exact_executable_coverage": sum(result["exact_executable"] for result in results) / len(results),
        "unsupported_reason_counts": dict(sorted(unsupported_reasons.items())),
        "preparation_failure_count": sum(
            result["compiled_outcome"]["status"] == "PREPARATION_FAILURE" for result in results
        ),
        "compiler_deterministic_count": sum(result["compiler_deterministic"] for result in results),
        "baseline_round_trip_check_count": sum(
            result["baseline_round_trip_exact"] is not None for result in results
        ),
        "baseline_round_trip_exact_count": sum(
            result["baseline_round_trip_exact"] is True for result in results
        ),
        "silent_relaxation_count": sum(result["silent_relaxations"] for result in results),
    }


def _observations(
    contract,
    source_gate,
    source,
    static_review,
    specs,
    results,
    replay_exact,
    worker_count,
) -> dict[str, Any]:
    summary = _summary(results)
    nontrivial_by_dag = {
        dag_class: any(
            result["dag_class"] == dag_class
            and result["exact_executable"]
            and not result["template"].endswith("cores:0,1,2,3|splits:D0=4")
            for result in results
        )
        for dag_class in contract["dag_classes"]
    }
    observations = {
        "source_gate_hash_matches": source_gate["hash_matches"],
        "source_identified": source["identified"],
        "static_review_scope_match": static_review["scope_match"],
        "assignment_denominator_complete": len(results) == len(specs) == contract["expected_assignment_count"],
        "paired_candidate_set_exact": summary["paired_candidate_set_exact_count"] == len(specs),
        "compiled_reference_outcome_exact": summary["compiled_reference_outcome_exact_count"] == len(specs),
        "parallel_worker_count_exact": worker_count == contract["max_workers"],
        "downstream_serial_parallel_replay_exact": replay_exact,
        "compiler_deterministic": summary["compiler_deterministic_count"] == len(specs),
        "baseline_round_trip_exact": summary["baseline_round_trip_check_count"]
        == summary["baseline_round_trip_exact_count"]
        == contract["expected_baseline_round_trip_count"],
        "nontrivial_exact_subset_every_dag": all(nontrivial_by_dag.values()),
        "unsupported_reasons_registered": set(summary["unsupported_reason_counts"])
        <= set(contract["allowed_unsupported_reasons"]),
        "unsupported_assignment_set_exact": sorted(
            result["assignment_id"] for result in results if not result["exact_executable"]
        )
        == contract["expected_unsupported_assignments"],
        "unsupported_reason_counts_exact": summary["unsupported_reason_counts"]
        == contract["expected_unsupported_reason_counts"],
        "exact_executable_count_matches": summary["exact_executable_count"]
        == contract["expected_exact_executable_count"],
        "unexpected_preparation_failure_count": summary["preparation_failure_count"],
        "silent_relaxation_count": summary["silent_relaxation_count"],
    }
    return observations


def _load_source_gate(contract) -> dict[str, Any]:
    expected = contract["source_gate"]
    path = Path(expected["artifact"])
    digest = _file_digest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = (
        digest == expected["sha256"]
        and payload.get("verdict") == expected["required_verdict"]
        and payload.get("evidence_class") == expected["required_evidence_class"]
    )
    if not valid:
        raise OperatorTemplateFaithfulnessError("Gate 2E-A source evidence does not match the frozen contract")
    return {
        "path": str(path),
        "sha256": digest,
        "verdict": payload["verdict"],
        "evidence_class": payload["evidence_class"],
        "hash_matches": True,
    }


def _validate_contract(contract: dict[str, Any]) -> None:
    expected = {
        "contract",
        "version",
        "source_gate",
        "dag_classes",
        "template_family",
        "assignment_policy",
        "reference_pipeline",
        "compile_repetitions",
        "expected_assignment_count",
        "expected_baseline_round_trip_count",
        "expected_exact_executable_count",
        "expected_unsupported_assignments",
        "expected_unsupported_reason_counts",
        "max_workers",
        "allowed_unsupported_reasons",
        "hardware",
        "static_review",
        "pass_criteria",
        "excluded_claims",
    }
    if set(contract) != expected or contract["version"] != _CONTRACT_VERSION:
        raise OperatorTemplateFaithfulnessError("invalid Gate 1A-v3 contract schema")
    paths = contract["static_review"]["expected_changed_paths"]
    if paths != sorted(set(paths)):
        raise OperatorTemplateFaithfulnessError("static-review paths must be sorted and unique")
    templates = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in contract["template_family"]]
    if not templates or len(templates) != len(set(templates)):
        raise OperatorTemplateFaithfulnessError("template family must be non-empty and unique")
    if any(count < 1 for count in contract["dag_classes"].values()):
        raise OperatorTemplateFaithfulnessError("DAG operator counts must be positive")
    operator_count = sum(contract["dag_classes"].values())
    if contract["expected_assignment_count"] != operator_count * len(templates):
        raise OperatorTemplateFaithfulnessError("assignment count is inconsistent with DAG and template cardinality")
    if contract["expected_baseline_round_trip_count"] != operator_count:
        raise OperatorTemplateFaithfulnessError("baseline round-trip count must equal the frozen operator count")
    unsupported = contract["expected_unsupported_assignments"]
    reason_counts = contract["expected_unsupported_reason_counts"]
    if unsupported != sorted(set(unsupported)) or sum(reason_counts.values()) != len(unsupported):
        raise OperatorTemplateFaithfulnessError("unsupported assignment denominator is inconsistent")
    if set(reason_counts) - set(contract["allowed_unsupported_reasons"]):
        raise OperatorTemplateFaithfulnessError("expected unsupported reason is not registered")
    if contract["expected_exact_executable_count"] + len(unsupported) != contract["expected_assignment_count"]:
        raise OperatorTemplateFaithfulnessError("executable and unsupported counts do not cover the denominator")
    if contract["compile_repetitions"] < _MIN_COMPILE_REPETITIONS or contract["max_workers"] <= 0:
        raise OperatorTemplateFaithfulnessError("repetition and worker counts must be positive and nontrivial")
    expected_criteria = {
        "source_gate_hash_matches",
        "source_identified",
        "static_review_scope_match",
        "assignment_denominator_complete",
        "paired_candidate_set_exact",
        "compiled_reference_outcome_exact",
        "parallel_worker_count_exact",
        "downstream_serial_parallel_replay_exact",
        "compiler_deterministic",
        "baseline_round_trip_exact",
        "nontrivial_exact_subset_every_dag",
        "unsupported_reasons_registered",
        "unsupported_assignment_set_exact",
        "unsupported_reason_counts_exact",
        "exact_executable_count_matches",
        "unexpected_preparation_failure_count",
        "silent_relaxation_count",
    }
    if set(contract["pass_criteria"]) != expected_criteria:
        raise OperatorTemplateFaithfulnessError("pass criteria are missing, unknown, or unevaluated")
    source_digest = contract["source_gate"]["sha256"]
    if not isinstance(source_digest, str) or not source_digest.startswith("sha256:"):
        raise OperatorTemplateFaithfulnessError("source-gate digest must use the sha256 namespace")


def _source_manifest(
    expected_commit: str | None,
    *,
    extra_snapshot_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    root_ok, root = _git_checked("rev-parse", "--show-toplevel")
    resolved_root = Path(root).resolve() if root_ok else None
    inside = resolved_root == Path.cwd().resolve()
    head_ok, head = _git_checked("rev-parse", "HEAD") if inside else (False, "")
    status_ok, status = _git_checked("status", "--porcelain") if inside else (False, "")
    module = Path(__file__).resolve()
    commit_matches = expected_commit is None or expected_commit == head
    return {
        "commit": expected_commit or head,
        "head": head or None,
        "git_root": root or None,
        "identified": bool(
            inside
            and head_ok
            and status_ok
            and module.is_relative_to(resolved_root)
            and commit_matches
            and not status
        ),
        "clean": not status if inside else None,
        "dirty_paths": status.splitlines(),
        "snapshot_digest": _source_snapshot_digest(extra_snapshot_paths),
        "expected_commit_matches_head": commit_matches,
    }


def _source_run_manifest(before, after, destination: Path) -> dict[str, Any]:
    roots = {item["git_root"] for item in (before, after) if item["git_root"]}
    stable = (
        len(roots) == 1
        and before["head"] == after["head"]
        and before["snapshot_digest"] == after["snapshot_digest"]
        and before["dirty_paths"] == after["dirty_paths"]
    )
    root = Path(next(iter(roots))).resolve() if len(roots) == 1 else None
    output_outside = root is not None and not destination.is_relative_to(root)
    return {
        **after,
        "identified": bool(before["identified"] and after["identified"] and stable and output_outside),
        "stable_during_run": stable,
        "output_outside_checkout": output_outside,
    }


def _static_review_manifest(contract, source) -> dict[str, Any]:
    expected = contract["static_review"]
    ok, changed = _git_checked("diff", "--name-only", f"{expected['base_commit']}..{source['commit']}")
    paths = sorted(path for path in changed.splitlines() if path)
    expected_paths = expected["expected_changed_paths"]
    return {
        "changed_paths": paths,
        "expected_changed_paths": expected_paths,
        "scope_match": ok and paths == expected_paths,
    }


def _environment_manifest() -> dict[str, Any]:
    installed_distributions = sorted(
        (distribution.metadata["Name"].lower(), distribution.version)
        for distribution in distributions()
        if distribution.metadata["Name"]
    )
    return {
        "host": platform.node(),
        "python": sys.version,
        "packages": {
            name: _package_version(name)
            for name in ("onnx", "ortools", "pyyaml", "stream-dse", "xdsl", "zigzag-dse")
        },
        "installed_distributions": [
            {"name": name, "version": package_version} for name, package_version in installed_distributions
        ],
        "installed_distribution_manifest_sha256": _digest(installed_distributions),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _source_snapshot_digest(extra_paths: tuple[Path, ...] = ()) -> str:
    entries = []
    roots = (Path("stream"), Path("scripts/run_operator_template_faithfulness.py"), *extra_paths)
    for root in roots:
        candidates = root.rglob("*") if root.is_dir() else (root,)
        for path in candidates:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json"}:
                entries.append((str(path), _file_digest(path)))
    return _digest(sorted(entries))


def _git_checked(*arguments: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(("git", *arguments), check=True, capture_output=True, text=True)
        return True, result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False, ""


def _file_digest(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = [
    "OperatorTemplateFaithfulnessError",
    "load_operator_template_faithfulness_contract",
    "run_operator_template_faithfulness",
]
