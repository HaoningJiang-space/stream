#!/usr/bin/env python3
"""Run the frozen Gate 2A real-workload prepare-only census."""

from __future__ import annotations

import argparse

from stream.structural.real_workload_lifting import run_real_workload_lifting


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/gate2a/report.json")
    parser.add_argument(
        "--source-commit",
        help="expected HEAD for an already-clean Git checkout; defaults to the current Git HEAD",
    )
    args = parser.parse_args()
    report = run_real_workload_lifting(args.output, source_commit=args.source_commit)
    summary = report["summary"]
    print(
        f"Gate 2A: {report['verdict']}; lifting="
        f"{summary['valid_workloads']}/{summary['required_workloads']}; "
        f"TTA constructed={summary['tta_constructed']}"
    )
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
