#!/usr/bin/env python3
"""Run Gate 2D alternative-route domain expansion."""

from __future__ import annotations

import argparse

from stream.structural.route_option_gate import run_route_option_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/gate2d/report.json")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    report = run_route_option_gate(args.output, source_commit=args.source_commit)
    expanded = report["results"][str(report["contract"]["expansion"]["expanded_limit"])]
    tensors = sum(result["opportunity"]["eligible_tensor_count"] for result in expanded.values())
    path_variables = sum(result["opportunity"]["path_nondegenerate_count"] for result in expanded.values())
    print(
        f"Gate 2D: {report['verdict']}; path_variables={path_variables}/{tensors}; "
        f"wall={report['execution']['wall_seconds']:.2f}s"
    )
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
