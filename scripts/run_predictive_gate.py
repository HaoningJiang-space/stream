#!/usr/bin/env python3
"""Run the frozen Gate 1B complete TETRA oracle census."""

from __future__ import annotations

import argparse

from stream.structural.predictive_gate import run_predictive_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/gate1b/report.json")
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--oci-source-digest", required=True)
    args = parser.parse_args()
    report = run_predictive_gate(
        args.output,
        container_image=args.container_image,
        oci_source_digest=args.oci_source_digest,
    )
    print(
        f"Gate 1B: {report['verdict']}; coverage={report['evaluation_coverage']:.3f}; "
        f"passing_classes={report['passing_dag_classes']}/{report['required_passing_dag_classes']}"
    )


if __name__ == "__main__":
    main()
