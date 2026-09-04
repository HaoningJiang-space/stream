"""Run the frozen Gate 1A conformance census."""

from __future__ import annotations

import argparse

from stream.structural.gate1a import run_gate1a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/gate1a/report.json")
    args = parser.parse_args()
    census = run_gate1a(args.output)
    print(f"Gate 1A: {census.verdict.value}; coverage={census.coverage:.3f}; assignments={len(census.audits)}")


if __name__ == "__main__":
    main()
