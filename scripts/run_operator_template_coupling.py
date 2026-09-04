#!/usr/bin/env python3
"""Run Gate 1A-v4 shared-tensor operator-template compatibility."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys

from stream.structural.operator_template_coupling import (
    run_operator_template_coupling,
    verify_operator_template_coupling_provenance,
    write_operator_template_coupling_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/stream-gate1a-v4-report.json")
    parser.add_argument("--source-commit")
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()
    invocation = (sys.executable, *sys.argv)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        report = run_operator_template_coupling(
            args.output,
            source_commit=args.source_commit,
            max_workers=args.max_workers,
            invocation=invocation,
        )
    summary = (
        f"Gate 1A-v4: {report['verdict']}/{report['classification']}; "
        f"assignments={report['summary']['assignment_count']}; "
        f"executable={report['summary']['exact_executable_count']}\n"
    )
    stdout_text = stdout.getvalue() + summary
    stderr_text = stderr.getvalue()
    manifest_path = write_operator_template_coupling_provenance(
        args.output,
        report,
        invocation,
        stdout_text,
        stderr_text,
    )
    if not verify_operator_template_coupling_provenance(manifest_path):
        raise RuntimeError("Gate 1A-v4 provenance bundle verification failed")
    print(stdout_text, end="")
    print(stderr_text, end="", file=sys.stderr)
    if report["verdict"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
