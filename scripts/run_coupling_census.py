#!/usr/bin/env python3
"""Run Gate 2C over the accepted Gate 2B opportunity artifact."""

from __future__ import annotations

import argparse

from stream.structural.coupling_census import run_coupling_census


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Gate 2B report; defaults to the contract-pinned artifact")
    parser.add_argument("--output", default="artifacts/gate2c/report.json")
    args = parser.parse_args()
    report = run_coupling_census(args.output, source_report=args.input)
    summary = report["summary"]
    print(
        f"Gate 2C: {report['verdict']}/{report['inference_regime']}; "
        f"variables={summary['variable_count']}; non-unary={summary['non_unary_factor_count']}; "
        f"primal-edges={summary['primal_graph_edge_count']}; width={summary['induced_width']}"
    )


if __name__ == "__main__":
    main()
