# Structural Predictive Value: Gate 1B

Gate 1B starts only after the frozen Gate 1A-v2 artifact reports `PASS`. It asks whether an integer structural score can retain low-latency TETRA solutions within a small candidate budget. It does not evaluate VE scalability or claim an end-to-end speedup over STREAM.

## Frozen Candidate Space

The census covers six DAG classes: edge, chain, GEMM chain, Conv chain, residual diamond, and fork-join. Each case exposes exactly three transfer-produced staging tensors. Every candidate selects one existing placement and one compatible existing path for every adjacent transfer of each tensor. The full Cartesian product contains

$$
4^3 = 64
$$

candidates per class, so `Top-16` does not saturate the search space. Only Gate 1A-v2-proven staging-placement and adjacent-path restrictions are admitted. `PARTIAL`, `STREAMING`, exact reuse, layout, VE, A*, beam search, and LNS remain excluded.

## Preregistered Structural Objective

For every selected transfer plan, the score charges each physical link-service event exactly once:

$$
J_{struct}(x)=
\sum_{e\in Events(x)}
\left\lceil
\frac{bits(e)}{BW(e)\,chains(e)}
\right\rceil.
$$

The score is integer-valued and may not inspect TETRA solutions, TETRA latency, timeslots, reuse choices, or solver objectives. Equal scores are ordered by the canonical SHA-256 candidate identifier. The machine-readable contract is `stream/structural/contracts/gate1b_contract.json`.

## Complete Oracle Census

For each class, the harness constructs a fresh unrestricted baseline and 64 freshly compiled restricted problems. Every problem uses the same workload class, accelerator, cost LUT, resource-aware timeslot policy, enabled constraint families, backend, solver parameters, time-limit policy, and accepted-status policy. Each restricted TTA domain must exactly equal its singleton structural literals. Evaluation is valid only when the baseline and all 64 candidates return `OPTIMAL`, domain checks pass, and invariant semantic manifests match.

The accepted run uses OR-Tools 9.15 or newer. Because eex004 has glibc 2.17, the run uses a hash-pinned Singularity image containing Debian 12 glibc and Python 3.12 plus an independent Gate 1B virtual environment. This changes no STREAM/TTA semantics and avoids modifying the host libc or weakening the dependency requirement. Image, OCI source, package-freeze, source, and Gate 1A-v2 artifact hashes are recorded in the result.

## Metrics and Verdict

The artifact reports `Recall@N` for $N\in\{1,4,8,16\}$ against TETRA Top-1 and Top-5, `Regret@N`, and Spearman correlation. Latency ties expand the good-solution set at the Top-$k$ cutoff; recall measures whether the selected budget retains at least $k$ equally good solutions. The decisive per-class criteria are

$$
Recall@16(Top5)\ge 0.8,
\qquad
Regret@16\le 0.05.
$$

Gate 1B passes only with 100% valid evaluation coverage and at least five of six passing DAG classes. The baseline is included in the final candidate set only as a non-regression sanity check. Before the first latency census, the status is `NOT_RUN`; the objective, denominator, metrics, and thresholds must not change after results are observed.
