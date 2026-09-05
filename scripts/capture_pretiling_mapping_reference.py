"""Capture the MappingParser-seam reference from an accepted historical checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _digest_file(path: Path) -> str:
    return "sha256:" + sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    ).stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _core_ids(option) -> list[int]:
    return [core.id for core in option]


def _mapping_manifest(workload, mapping) -> dict[str, Any]:
    nodes = {}
    for node in sorted(workload.get_computation_nodes(), key=lambda item: item.name):
        node_mapping = mapping.get(node)
        nodes[node.name] = {
            "resource_options": [_core_ids(option) for option in node_mapping.resource_allocation],
            "tiling_options": [
                [{"position": dimension.position, "factor": factor} for dimension, factor in option]
                for option in node_mapping.inter_core_tiling
            ],
            "memory_options": [_core_ids(option) for option in node_mapping.memory_allocation],
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


def _run_stage(stage, context):
    from stream.stages.stage import LeafStage, MainStage

    results = MainStage([stage, LeafStage], context).run()
    if len(results) != 1:
        raise RuntimeError(f"{stage.__name__} returned {len(results)} contexts")
    return results[0]


def capture(source_root: Path, output: Path, expected_commit: str) -> dict[str, Any]:
    source_root = source_root.resolve()
    output = output.resolve()
    instrument_path = Path(__file__).resolve()
    instrument_root = instrument_path.parents[1]
    instrument_head = _git(instrument_root, "rev-parse", "HEAD")
    instrument_status = _git(instrument_root, "status", "--porcelain")
    head = _git(source_root, "rev-parse", "HEAD")
    status = _git(source_root, "status", "--porcelain")
    if instrument_status:
        raise RuntimeError("reference instrument must come from a clean commit")
    if head != expected_commit or status:
        raise RuntimeError("reference source must be the expected clean commit")
    if output.is_relative_to(source_root):
        raise RuntimeError("reference output must be outside the source checkout")

    sys.path.insert(0, str(source_root))
    from stream.stages.context import StageContext
    from stream.stages.generation.generic_mapping_generation import GenericMappingGenerationStage
    from stream.stages.generation.normalization_expansion import ExpandNormalizationStage
    from stream.stages.parsing.accelerator_parser import AcceleratorParserStage
    from stream.stages.parsing.mapping_parser import MappingParserStage
    from stream.stages.parsing.onnx_model_parser import ONNXModelParserStage

    contract_path = source_root / "stream/structural/contracts/gate2a_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    original_cwd = Path.cwd()
    workloads = {}
    try:
        os.chdir(source_root)
        with TemporaryDirectory(prefix="stream-gate2fa-reference-") as temporary:
            temporary_root = Path(temporary)
            for workload_spec in contract["workloads"]:
                context = StageContext.from_kwargs(
                    accelerator=contract["hardware"],
                    workload_path=workload_spec["path"],
                    output_path=str(temporary_root / workload_spec["id"]),
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
                groups = []
                for index, (workload, mapping_path) in enumerate(
                    zip(context.get("sub_workloads"), context.get("group_mapping_paths"), strict=True)
                ):
                    group_context = StageContext.from_kwargs(
                        accelerator=context.get("accelerator"),
                        workload=workload,
                        mapping_path=mapping_path,
                        output_path=str(temporary_root / workload_spec["id"] / f"group_{index}"),
                    )
                    group_context = _run_stage(MappingParserStage, group_context)
                    groups.append(
                        {
                            "operator_ids": [node.name for node in workload.get_computation_nodes()],
                            "mapping": _mapping_manifest(workload, group_context.get("mapping")),
                            "mapping_file_sha256": _digest_file(Path(mapping_path)),
                        }
                    )
                workload_path = source_root / workload_spec["path"]
                workloads[workload_spec["id"]] = {
                    "family": workload_spec["family"],
                    "sha256": _digest_file(workload_path),
                    "frontend_trace": trace,
                    "group_trace": ["mapping_parser"],
                    "groups": groups,
                }
    finally:
        os.chdir(original_cwd)

    post_head = _git(source_root, "rev-parse", "HEAD")
    post_status = _git(source_root, "status", "--porcelain")
    if post_head != head or post_status:
        raise RuntimeError("reference source changed during capture")
    return {
        "schema": "gate2f-a-pretiling-reference-v1",
        "source": {"commit": head, "clean": True},
        "instrument": {
            "commit": instrument_head,
            "path": str(instrument_path.relative_to(instrument_root)),
            "sha256": _digest_file(instrument_path),
        },
        "environment": {
            "python_version": ".".join(str(component) for component in sys.version_info[:3]),
            "packages": {
                name: _package_version(name)
                for name in ("onnx", "ortools", "stream-dse", "zigzag-dse")
            },
        },
        "hardware": {
            "path": contract["hardware"],
            "sha256": _digest_file(source_root / contract["hardware"]),
        },
        "workloads": workloads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    report = capture(args.source_root, args.output, args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PRETILING_REFERENCE_CAPTURED {args.output}")


if __name__ == "__main__":
    main()
