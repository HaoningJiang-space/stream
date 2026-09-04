# Gate 2C Result: Executable Factor Coupling Census

## Verdict

Gate 2C is **PASS / SEPARABLE**. The accepted eex004 run used clean commit
`273a69ec6d30a9da2d824266ecc0de8f01ecbece` and the pinned Gate 2B artifact. The result is
[`artifacts/gate2c/report.json`](../../artifacts/gate2c/report.json), SHA-256:

```text
18a002c41a69397596ef4ca5783d2ebc7e647d85950fc0dd4f1ea269f7a32d79
```

## Claim-Evidence Matrix

| Claim | Evidence | Result | Boundary |
|---|---|---:|---|
| Current executable structural proxy is separable | Factor arities and primal edges | 17 unary, 0 non-unary, 0 edges | Structural proxy only |
| Independent argmin is exact for this proxy | Induced width and domain sums | width 0; 85 local evaluations | No TETRA feasibility claim |
| VE is not currently needed | Largest coupled component | 1 | No VE speedup was measured |
| Opportunity remains narrow | Gate 2B handoff | placement-only; 2.307% logical bits | Logical coverage is not NoC traffic |

## Per-Workload Structure

| Workload | Variables | Naive assignments | Exact local evaluations | Non-unary factors |
|---|---:|---:|---:|---:|
| SwiGLU | 1 | 5 | 5 | 0 |
| FSRCNN | 8 | 390,625 | 40 | 0 |
| ResNet18 | 6 | 15,625 | 30 | 0 |
| Attention head | 2 | 25 | 10 | 0 |

The frozen score has the form

\[
J_{\mathrm{struct}}(x)=\sum_t J_t(x_t).
\]

Every link-service event belongs to one tensor-owned factor, so the primal graph has no edges. The
exact optimum is therefore

\[
x_t^*=\arg\min_{x_t\in D_t}J_t(x_t),
\]

requiring \(\sum_t|D_t|=85\) local evaluations across the four separate workload censuses. The
products shown above are domain-size diagnostics, not work performed by the correct solver and not
evidence that VE scales to a large coupled problem.

## Decision

Do not optimize VE, its elimination order, A*, beam search, or LNS for the current executable
space. The next research step is an option-generation audit. Large singleton tensors dominate the
eligible logical volume, and each fixed placement currently has only one compatible path tuple.
The audit should determine why production mapping generates singleton placement domains for these
tensors and singleton route domains for fixed endpoints. Only after a sound domain extension should
capacity, contention, shared-buffer, reuse, partial-materialization, or streaming factors be added
to create physically meaningful cross-tensor coupling.

TETRA itself is not claimed to be separable: memory capacity, DMA, and link contention remain in
the unchanged downstream model and are outside this structural factor census.
