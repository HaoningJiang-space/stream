#!/usr/bin/env python3
"""Run Gate 1A-v3 paired operator-template compiler faithfulness."""

from __future__ import annotations

import argparse

from stream.structural.operator_template_faithfulness import run_operator_template_faithfulness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/stream-gate1a-v3-report.json")
    parser.add_argument("--source-commit")
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    report = run_operator_template_faithfulness(
        args.output,
        source_commit=args.source_commit,
        max_workers=args.max_workers,
    )
    print(
        f"Gate 1A-v3: {report['verdict']}; assignments={report['summary']['assignment_count']}; "
        f"executable={report['summary']['exact_executable_count']}"
    )
    if report["verdict"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
