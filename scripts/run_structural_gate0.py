"""Run the exact tensor-realization structural Gate 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream.structural.gate0 import run_gate0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-per-class", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, default=Path("outputs/structural_gate0/report.json"))
    args = parser.parse_args()

    report = run_gate0(args.configs_per_class, args.base_seed)
    report.write_json(args.output)
    print(json.dumps(report.summary(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
