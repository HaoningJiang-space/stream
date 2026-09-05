# Gate 2F-A Result: Pre-Tiling Operator Opportunity

## Verdict

Gate 2F-A is **DISCOVERY_PASS**. It establishes that the frozen real-workload mapping space has nontrivial paired `(core group, local split)` templates and that its pre-tiling shared-tensor signature proxy is non-Cartesian. It does not establish post-tiling compatibility, TTA feasibility, latency improvement, or a need for variable elimination.

## Correctness Evidence

Two independent eex004 runs used the isolated Python 3.12.11 / OR-Tools 9.15.6755 environment and a clean public checkout. Both completed under the 120-second contract budget (7.00 s and 7.06 s). Ruff and the focused pre-tiling unit suite passed before the census.

All 19 frozen correctness criteria passed in both runs: the Gate 2A workload and mapping-parser reference matched, the allowlisted stage trace stopped at `MappingParser`, no forbidden execution occurred, every generated template compiled and was unique, every baseline template was retained, reduction-axis splitting was zero, and shared-tensor factors were fully covered with analytic counts matching direct enumeration.

Each outer run performs two preparations per workload. The two outer reports have identical semantic content after removing only timestamps and wall-clock measurements; their provenance bundles verify independently. The compressed first report, its provenance manifest, and the repeat audit are in [`artifacts/gate2f-a/`](../../artifacts/gate2f-a/). Decode the raw report with `base64 -d report.json.gz.b64 | gzip -dc`.

## Opportunity Census

| Workload family | Operators | Templates | Shared factors | Non-Cartesian factors |
|---|---:|---:|---:|---:|
| Transformer FFN (SwiGLU) | 5 | 382 | 1 | 1 |
| Sequential CNN (FSRCNN) | 8 | 1,533 | 1 | 1 |
| Residual CNN (ResNet18) | 48 | 3,423 | 8 | 6 |
| Attention fork-join | 9 | 636 | 3 | 3 |
| **Total** | **70** | **5,974** | **13** | **11** |

Every operator has a non-singleton template domain (7--217 choices). Non-Cartesian factors occur in all four frozen workload families, exceeding the discovery threshold of three.

## Next Gate

The signature relation is still a pre-tiling proxy. Gate 2F-B must test whether each proxy-compatible tuple preserves the same shared-tensor relation through `TilingGeneration`, transfer-graph construction, SSIS, and mapping update. Only a surviving post-tiling relation can justify component DP/VE or an approximate structural search.
