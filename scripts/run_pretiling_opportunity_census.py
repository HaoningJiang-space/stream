#!/usr/bin/env python3
"""Run Gate 2F-A at the production pre-tiling compiler seam."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys

from stream.structural.pretiling_opportunity_census import (
    run_pretiling_opportunity_census,
    verify_pretiling_opportunity_provenance,
    write_pretiling_opportunity_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/stream-gate2f-a-report.json")
    parser.add_argument("--source-commit")
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    invocation = (sys.executable, *sys.argv)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        report = run_pretiling_opportunity_census(
            args.output,
            source_commit=args.source_commit,
            max_workers=args.jobs,
        )
    summary = report["summary"]
    summary_line = (
        f"Gate 2F-A: {report['verdict']}; operators={summary['operator_count']}; "
        f"templates={summary['candidate_state_count_sum']}; "
        f"shared factors={summary['shared_tensor_factor_count']}; "
        f"non-Cartesian={summary['noncartesian_factor_count']}; "
        f"wall={report['execution']['wall_seconds']:.2f}s\n"
    )
    stdout_text = stdout.getvalue() + summary_line
    stderr_text = stderr.getvalue()
    manifest_path = write_pretiling_opportunity_provenance(
        args.output,
        report,
        invocation,
        stdout_text,
        stderr_text,
    )
    if not verify_pretiling_opportunity_provenance(manifest_path):
        raise RuntimeError("Gate 2F-A provenance bundle verification failed")
    print(stdout_text, end="")
    print(stderr_text, end="", file=sys.stderr)
    if report["run_status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
