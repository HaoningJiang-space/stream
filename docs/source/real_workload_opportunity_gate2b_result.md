# Gate 2B Result: Real-Workload Opportunity Census

## Verdict

Gate 2B is **NARROW / PLACEMENT_ONLY**. The census ran on eex004 from clean commit
`0ed67c6de86c413c25a3ca65b4459d0af64ba40e` and consumed the accepted Gate 2A artifact with
SHA-256 `d2473721f8e9d4796db5db7797ddf7c0099e82d4788ab4d081fe9b6a218aa145`. The result artifact is
[`artifacts/gate2b/report.json`](../../artifacts/gate2b/report.json), SHA-256:

```text
49a7a09edcdbae137c72ed5ffdfc2bf3b021d9809ab71d2f36ba1584043bf971
```

## Census Results

| Workload | Eligible tensors | Non-degenerate | Naive assignment space | Controlled logical bits |
|---|---:|---:|---:|---:|
| SwiGLU | 5 | 1 | 5 | 0.15% |
| FSRCNN | 9 | 8 | 390,625 | 3.15% |
| ResNet18 | 43 | 6 | 15,625 | 3.79% |
| Attention head | 8 | 2 | 25 | 25.81% |
| **Portfolio** | **65** | **17** | not combined across workloads | **2.31%** |

For a staging tensor (t), the exact executable domain is reconstructed as

\[
|D_t|=\sum_{p\in P_t}\prod_{e\in Adj(t)}|R_e(p)|.
\]

A tensor is non-degenerate when \(|D_t|>1\). Logical bits are tensor elements multiplied by the
fixed element bitwidth recorded in the Gate 2A manifest. They are a coverage measure, not physical
NoC traffic: iteration multiplicity, reuse, overlap, and contention are intentionally excluded.

## Interpretation

The raw Cartesian products are non-trivial, particularly FSRCNN and ResNet18, but they do not yet
represent a broad tensor-realization search. All 17 non-degenerate variables have five compatible
placement choices. For every fixed placement, every adjacent transfer has exactly one compatible
path; therefore the number of independently selectable path variables is zero. Large products such
as \(5^8\) must not be presented as evidence of cross-tensor coupling or VE scalability.

The controlled-data result further limits the opportunity claim. Non-degenerate tensors account for
only 19,027,360 of 824,874,400 eligible logical bits (2.307%). Much of the apparent combinatorial
space comes from small inputs or weights, while several large activation and weight tensors retain
singleton domains.

## Decision

Gate 2B establishes a real but narrow placement opportunity. The next audit is Gate 2C factor
coupling: count non-unary factors and primal-graph edges in the exact executable model. If it is
fully unary as predicted, independent per-tensor argmin is the correct solver; extending tensor
semantics and option generation takes priority over VE, A*, beam search, or LNS.
