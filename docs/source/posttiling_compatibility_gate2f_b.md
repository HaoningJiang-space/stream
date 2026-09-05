# Gate 2F-B: Post-Tiling Compatibility Faithfulness

Gate 2F-A found non-Cartesian shared-tensor factors at the `MappingParser` seam. Gate 2F-B tests whether that pre-tiling relation is identical to the relation enforced by STREAM after operator-template compilation, tiling, transfer-graph construction, mapping update, and SSIS generation.

For every concrete consumer-template tuple in every frozen Gate 2F-A factor, non-factor operators retain their parsed baseline mappings. The tuple is classified by

\[
R_{pre}=\left|\{s_i:s_i\ne\varnothing\}\right|\le 1,
\]

where each \(s_i\) is Gate 2F-A's tensor-relevant signature. `R_post` is taken from the production transfer-mapping path. Each generated transfer hop carries immutable source-tensor lineage, and production records the consumer projections, selected reference, result tiling, and option-domain sizes. A structured shared-input incompatibility is `R_post = 0`; successful mapping plus SSIS generation with complete lineage witnesses is `R_post = 1`.

The gate reports false positives (`R_pre = 1`, `R_post = 0`) and false negatives (`R_pre = 0`, `R_post = 1`) separately. `PASS` requires both counts to be zero.

## Correctness Boundary

Every ordered template tuple is enumerated. Domain keys and counts must match the accepted Gate 2F-A artifact, selected core/split literals must survive every transformation, and tuple accounting must satisfy

\[
N_{expected}=N_{enumerated}=N_{valid}.
\]

Any source drift, missing or ambiguous lineage, lost literal, empty production domain, unexpected exception, incomplete stage trace, or nondeterministic replay makes the entire run `INVALID`; such rows never enter the confusion matrix. Runtime auditing forbids TTA construction/solve, exhaustive structural optimization, and variable elimination. Tiled-workload visualization is disabled because it is nonsemantic and would otherwise dominate tuple-enumeration time.

The production communication manager memoizes repeated path-planning requests with identical ordered source
cores, destination cores, and plan limits. This changes no domain or trace: each concrete tuple still traverses
the complete pipeline, while identical deterministic routing subproblems are reused.

Gate 2F-B does not establish TTA feasibility, performance improvement, factor-graph width, or the need for a particular search algorithm.
