#!/usr/bin/env python3
"""Run Gate 2B over the accepted Gate 2A real-workload artifact."""

from __future__ import annotations

import argparse

from stream.structural.opportunity_census import run_opportunity_census


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Gate 2A report; defaults to the contract-pinned artifact")
    parser.add_argument("--output", default="artifacts/gate2b/report.json")
    args = parser.parse_args()
    report = run_opportunity_census(args.output, source_report=args.input)
    summary = report["summary"]
    print(
        f"Gate 2B: {report['verdict']}/{report['opportunity_class']}; "
        f"nondegenerate={summary['nondegenerate_tensor_count']}/{summary['eligible_tensor_count']}; "
        f"controlled_bits={summary['controlled_logical_bits_ratio']:.3%}; "
        f"path_variables={summary['path_nondegenerate_count']}"
    )
    if report["verdict"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
