# Gate 1B Result: Structural Predictive Value

Gate 1B is `PASS` under the preregistered contract at commit `25beed6`. The accepted eex004 run evaluated all 64 concrete staging-placement/path assignments for each of six DAG classes with the unchanged TTA objective and constraints.

## Accepted Result

| DAG class | Evaluated | Recall@16 (Top-5) | Regret@16 | Spearman | Best TTA cycles | Baseline cycles |
|---|---:|---:|---:|---:|---:|---:|
| Edge | 64 / 64 | 1.00 | 0.00 | 0.960 | 24 | 25 |
| Chain | 64 / 64 | 1.00 | 0.00 | 0.971 | 35 | 35 |
| GEMM chain | 64 / 64 | 1.00 | 0.00 | 0.971 | 63 | 63 |
| Conv chain | 64 / 64 | 1.00 | 0.00 | 0.971 | 46 | 46 |
| Residual diamond | 64 / 64 | 1.00 | 0.00 | 0.971 | 52 | 52 |
| Fork-join | 64 / 64 | 1.00 | 0.00 | 0.971 | 68 | 68 |

All 384 candidates and six baselines returned `OPTIMAL`; every singleton placement/path domain matched its structural restriction, and every within-class evaluation-invariant hash matched its baseline. Six of six classes meet the required `Recall@16(Top-5) >= 0.8` and `Regret@16 <= 0.05`, exceeding the five-class pass threshold. Structural Top-8 already attains Top-5 recall 1.00 and zero regret in every class. Structural Top-1 has zero regret in every class.

## Independent Audit

An independent standard-library audit recomputed every structural score from its physical link-event ledger, checked unique event ownership, recomputed Recall, Regret, and tied-rank Spearman from the raw 384 evaluations, and verified all 19 recorded source hashes. It also confirmed the frozen Gate 1A-v2 input artifact hash. The remote SIF and `pip freeze` hashes match the artifact exactly.

The accepted environment used Python 3.12.11, glibc 2.36 inside Singularity, OR-Tools 9.15.6755, and the `ORTOOLS_GSCIP` backend. This is a user-space compatibility layer over the eex004 host; no host libc, TTA objective, timeslot policy, or constraint semantics were changed.

The evidence artifact is `artifacts/gate1b/report.json` (SHA-256 `45601a0a6c7854bbefc80bdd2aebd0462c186484e4bd4afdac7a7917a81230ae`). It contains the full candidate census, baseline results, metric inputs and outputs, event ledgers, restrictions, solver status, environment manifest, and source provenance.

## Evidence Boundary

The result establishes that `total_link_service_cycles` is an effective ranking proxy for the frozen 64-candidate staging-placement/path space on these six synthetic micro/meso DAGs. It does not establish general-network scalability, benefit from VE, support for `PARTIAL` or `STREAMING`, or a broad latency improvement over STREAM. Five baselines are already TTA-optimal in this space; only the edge case improves, from 25 to 24 cycles. Therefore Gate 1B validates candidate retention, not end-to-end performance superiority. The next admissible stage is a separately frozen scalability census on real workload graphs.
