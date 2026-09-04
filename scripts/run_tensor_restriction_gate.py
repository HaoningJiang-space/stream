"""Run the preregistered Minimal Tensor Restriction Gate 1A-v2."""

from __future__ import annotations

import argparse

from stream.structural.tensor_restriction_gate import run_tensor_restriction_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/gate1a-v2/report.json")
    args = parser.parse_args()
    report = run_tensor_restriction_gate(args.output)
    verdict = report["verdict"]
    coverage = report["coverage"]
    assignment_count = report["assignment_count"]
    summary = f"Gate 1A-v2: {verdict}; coverage={coverage:.3f}; assignments={assignment_count}"
    print(summary)


if __name__ == "__main__":
    main()
