# Gate 2E-A: Operator-Template Opportunity Census

Gate 2E-A is **DISCOVERY PASS / POTENTIAL ONLY**. A bounded library of output-indexed inter-core splits and compute-group templates exposes a large finite state space on every frozen real workload. This is an opportunity result, not an executable-compiler result.

## Frozen Template Rules

For each operator, the census uses only local iteration dimensions referenced by its output affine maps. A split factor must divide that dimension's extent; reduction-axis splits are excluded. The product of split factors must divide the size of the operator's single baseline compute pool. For a product of $p$, core groups are non-overlapping contiguous chunks of $p$ cores from that sorted pool.

The existing baseline mapping remains mandatory for future evaluation but is excluded from opportunity counts until semantic equivalence with a generated template is proven.

## Accepted Result

The clean run used public commit `983d6db7a6ebddcab39569491ff0d5cc8e63d5d1` on eex004. Its artifact is `artifacts/gate2e-a/report.json`, SHA-256 `4fc553a1ccf04377261aa48103078942f4db595e03df8fe594e3a244aab8ce49`.

| Workload | Operators | Nondegenerate | Largest group $\log_{10}|X|$ | Potential tensor-coupling edges |
|---|---:|---:|---:|---:|
| SwiGLU | 5 | 5 | 8.320 | 4 |
| FSRCNN | 8 | 8 | 18.013 | 7 |
| ResNet18 | 48 | 48 | 10.428 | 37 |
| Attention | 9 | 9 | 12.090 | 9 |

Across all workloads, the library contains 5,873 local operator states. Each operator has 7–217 states, and all 70 operators are nondegenerate. The 57 compute-to-compute edges identify where selecting producer and consumer templates may change tensor fragmentation or endpoints and therefore induce pairwise factors.

The Cartesian sizes above are nominal per fusion group. They do not demonstrate solver scalability and must not be multiplied across independent groups or workloads.

## Evidence Boundary and Next Gate

The census reads the accepted Gate 2A production-lifting artifact and the negative Gate 2D route-opportunity artifact. It does not run `MappingParser`, `TilingGeneration`, transfer lifting, TTA, or VE for generated templates. Consequently it does not establish compiler exactness, feasibility, latency improvement, or actual factor width.

The next required step is **Gate 1A-v3: Operator-Template Compilation Faithfulness**. A generated `(split template, core group)` must be selected before `TilingGeneration`, compiled through the production path, and compared with an independently constructed singleton-mapping reference. Only templates passing soundness, completeness, deterministic round trip, and baseline retention may enter a performance or scalability experiment.
