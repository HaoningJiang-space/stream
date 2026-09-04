# Gate 2A Result: Real-Workload Lifting Validity

## Accepted Run

Gate 2A passed on `eex004` from the clean Git checkout at
`fdce77dffa545b21fa14472cefcf156f526931ed`. The accepted environment used Python 3.12.11,
ONNX 1.21.0, OR-Tools 9.15.6755, ZigZag 3.8.5, and STREAM 1.14.1. The normalized invocation was:

```bash
singularity exec --pwd /workspace \
  --env LD_LIBRARY_PATH=/host-libs \
  --bind "$CLEAN_CHECKOUT:/workspace" \
  --bind "$EXTERNAL_RESULTS:/results" \
  --bind "$REMOTE_VENVS:/venvs" \
  --bind "$HOST_GIT:/usr/bin/git" \
  --bind "$HOST_LIBPCRE:/host-libs/libpcre.so.1" \
  "$PYTHON_RUNTIME" \
  /venvs/stream-gate1b-ortools915/bin/python \
  /workspace/scripts/run_real_workload_lifting.py \
  --output /results/gate2a-report.json \
  --source-commit fdce77dffa545b21fa14472cefcf156f526931ed
```

The first clean-checkout attempt correctly returned `NOT_RUN` because the minimal runtime image did
not contain Git, so the fail-closed source check could not identify the checkout. The accepted rerun
only added read-only bindings for the host Git executable and its missing `libpcre` dependency; the
source commit, workloads, environment, contract, and preparation path were unchanged.

## Result

- Verdict: `PASS`
- Workloads valid: 4/4; isolated attempts: 8/8 `LIFT_OK`
- Deterministic semantic hashes: 4/4
- Complete provenance, stage traces, placement domains, and path domains: yes
- Declared semantic exclusions: 21 in ResNet18; zero in the other frozen workloads
- Forbidden execution counts: TTA construction 0, TTA solve 0, exhaustive search 0, VE 0
- Source: clean, exact commit matched, stable before/after, result written outside the checkout
- Remote regression suite: 664 passed, 10 skipped, 34 deselected

The immutable report is [`artifacts/gate2a/report.json`](../../artifacts/gate2a/report.json)
(4,373,310 bytes), SHA-256:

```text
d2473721f8e9d4796db5db7797ddf7c0099e82d4788ab4d081fe9b6a218aa145
```

## Claim Boundary

This result establishes deterministic production-path lifting validity for the frozen Gate 2A
workload and environment contract. It does not establish tensor-decision opportunity, factor
coupling, VE scalability, TETRA latency benefit, or any Gate 2B/2C/2D result.

## Runner Performance Update

Commit `f160e7bd31cd7f1cbcc5635c9543220842ef76a6` parallelizes the eight independent preparation
attempts without changing the frozen workload, repeat count, hash seeds, or preparation path. On
the same eex004 runtime, total wall time decreased from 135.12 seconds to 30.21 seconds (4.47x),
while the four workload semantic hashes remained bit-identical and Gate 2A passed again. The
optimized report records eight workers and 27.15 seconds in the preparation section.

The runner now also fails source identification when the imported Gate module is outside the clean
checkout. Formal invocations therefore set `PYTHONPATH=/workspace`. The earlier accepted run's
editable-install module and clean-checkout module had identical SHA-256 digests, but the new check
prevents a future byte mismatch from being silently accepted.
