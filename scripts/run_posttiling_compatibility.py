#!/usr/bin/env python3
"""Run Gate 2F-B against the production post-tiling compatibility relation."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from stream.structural.posttiling_compatibility import (  # noqa: E402
    run_posttiling_compatibility,
    verify_posttiling_compatibility_provenance,
    write_posttiling_compatibility_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/stream-gate2f-b-report.json")
    parser.add_argument("--source-commit")
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    invocation = (sys.executable, *sys.argv)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        report = run_posttiling_compatibility(
            args.output,
            source_commit=args.source_commit,
            max_workers=args.jobs,
        )
    summary = report["summary"]
    summary_line = (
        f"Gate 2F-B: {report['verdict']}; factors={summary['factor_count']}; "
        f"tuples={summary['valid_tuple_count']}/{summary['expected_tuple_count']}; "
        f"FP={summary['false_positive_count']}; FN={summary['false_negative_count']}; "
        f"wall={report['execution']['wall_seconds']:.2f}s\n"
    )
    stdout_text = stdout.getvalue() + summary_line
    stderr_text = stderr.getvalue()
    manifest_path = write_posttiling_compatibility_provenance(
        args.output,
        report,
        invocation,
        stdout_text,
        stderr_text,
    )
    if report["run_status"] == "COMPLETED" and not verify_posttiling_compatibility_provenance(manifest_path):
        raise RuntimeError("Gate 2F-B provenance bundle verification failed")
    print(stdout_text, end="")
    print(stderr_text, end="", file=sys.stderr)
    if report["run_status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
