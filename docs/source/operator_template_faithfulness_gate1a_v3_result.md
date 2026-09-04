# Gate 1A-v3: Operator-Template Compilation Faithfulness

## Verdict

Gate 1A-v3 is **NARROW**. The paired pre-tiling compiler is faithful for the frozen micro-model, while the current transfer semantics reject a specific shared-input subset.

The accepted run used commit `102556861e679e50575f73f61c2e503573e2a354` on `eex004`, Python 3.12.11, OR-Tools 9.15.6755, ZigZag 3.8.5, and STREAM 1.14.1. The report is stored at `artifacts/gate1a-v3/report.json` with SHA-256 `8e1fd5771393da2f62766037b14caac67b2207fee90ac7b42e60ef3373832716`.

## Result

The frozen denominator contains 112 assignments: every operator in six micro-DAG classes crossed with seven atomic `(core group, split template)` choices.

| Check | Result |
|---|---:|
| Paired candidate-set equality | 112 / 112 |
| Compiled/reference outcome equality | 112 / 112 |
| Exactly executable assignments | 104 / 112 (92.86%) |
| Baseline round trips | 16 / 16 |
| Serial versus 8-worker replay | Exact |
| Silent relaxations | 0 |
| Unexpected preparation failures | 0 |

All eight unsupported assignments reproduce the same registered `SHARED_INPUT_TILING_INCOMPATIBLE` condition in both the compiled and independently constructed reference arms. They arise when only one consumer of a shared tensor changes from a `z0 × 4` to a `z0 × 2` split.

## Interpretation and Next Gate

The result proves exact compilation only for this finite micro-model through transfer construction, option generation, SSIS, and timeslot preparation. It does not prove that all 5,873 Gate 2E-A potential states are executable, that multi-operator assignments are feasible, or that latency improves.

The failure pattern exposes a useful coupled decision: consumers sharing one input must choose compatible tensor-relevant splits under the current transfer IR. A joint assignment may therefore restore feasibility where either single-operator mutation is unsupported. The next gate should test paired consumer changes such as `B/C: z0 × 2` in fork-join and `A/Add: z0 × 2` in the residual diamond before lifting this compatibility factor to real workloads.
