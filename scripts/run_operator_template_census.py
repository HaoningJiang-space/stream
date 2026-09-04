#!/usr/bin/env python3
"""Run Gate 2E-A potential operator-template opportunity census."""

from __future__ import annotations

import argparse

from stream.structural.operator_template_census import run_operator_template_census


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/gate2e-a/report.json")
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    report = run_operator_template_census(args.output, source_commit=args.source_commit)
    print(
        f"Gate 2E-A: {report['verdict']}/{report['evidence_class']}; "
        f"operators={report['summary']['operator_count']}; "
        f"states={report['summary']['candidate_state_count_sum']}"
    )
    if report["verdict"] != "DISCOVERY_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
