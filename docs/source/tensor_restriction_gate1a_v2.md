# Minimal Tensor Restriction Extension: Gate 1A-v2

Gate 1A-v1 remains frozen at commit `0b7d358` with verdict `FAIL`: native STREAM could not express a nontrivial tensor-realization choice. Gate 1A-v2 does not revise that result. It evaluates STREAM plus one minimal, first-class restriction seam over option domains that TTA already generates.

## Exact Supported Contract

The v2 interface accepts immutable `TensorRestriction` objects containing:

- allowed placements for a transfer-produced staging tensor; and
- allowed path plans for every transfer adjacent to that tensor.

For a baseline TTA placement domain $P_t$ and path domain $R_e$, compilation performs only exact intersection:

$$
P_t' = P_t \cap P_t^{allowed}, \qquad
R_e' = R_e \cap R_e^{allowed}.
$$

Unknown, empty, duplicate, incomplete, or incompatible restrictions fail closed. Path choices are filtered in the post-transfer mapping before `get_timeslots()`, because timeslot construction reads path link usage. The same immutable contract reaches TTA, where the normalized option domains are checked and filtered before variables and constraints are built.

```text
build_transfer_graph() -> update_mapping()
  -> filter staging placement and adjacent paths
  -> update_cost_lut() -> generate_ssis()
  -> get_timeslots(filtered mapping)
  -> initialize TTA domains
  -> exact domain intersection
  -> unchanged TTA model
```

This extension does not change the TTA objective, capacity constraints, routing model, timeslot policy, backend, or reuse semantics.

## Explicitly Unsupported Semantics

The exact v2 claim does not include compute-output placement, `PARTIAL`, `STREAMING`, exact reuse, output-layout constraints, or an abstract `FULL`/`BLOCK`/`MULTICAST` state that lacks a unique physical concretization. In particular, streaming requires temporal producer-consumer overlap and cannot be represented by placement and path filtering alone.

## Conformance Method

The denominator is frozen in `stream/structural/contracts/gate1a_intended_space_v2.json`: six micro-DAG classes and six preregistered placement subsets form 36 coverage cells. Each baseline problem exposes four staging placements and four paths for each adjacent transfer.

For every cell, the harness constructs a fresh unrestricted STREAM problem and independently enumerates its canonical TTA semantic solutions. It then verifies:

1. compiled candidate domains equal the requested intersections;
2. every restricted solution belongs to the unrestricted set intersected with the structural literals (soundness);
3. every unrestricted solution satisfying those literals remains feasible (completeness); and
4. every allowed option has a feasible witness.

The determinism census contains 1,000 assignments, each compiled from a fresh scheduler three times. Empty-restriction compilation is also compared with the uninstrumented baseline by pipeline manifest, option domains, and complete bounded feasible sets.

## Result

Gate 1A-v2 is `PASS`:

| Check | Result |
|---|---:|
| Exact coverage cells | 36 / 36 |
| Deterministic assignments | 1,000 / 1,000 |
| Repeated hashes per assignment | 3 / 3 identical |
| False `EXACT` | 0 |
| Silent relaxation | 0 |
| Unintended restriction | 0 |
| Semantic failure | 0 |
| Baseline round-trip failure | 0 |
| Nondeterministic compilation | 0 |

The complete artifact is `artifacts/gate1a-v2/report.json`. It embeds the frozen v1 artifact hash, v2 source hashes, environment metadata, all cell evidence, and all 1,000 determinism proofs. The accepted run completed on eex004 in 15 minutes 11 seconds.

The remote environment used Python 3.12.3 and OR-Tools 9.11 on glibc 2.17, while this repository declares OR-Tools 9.15 or newer. The finite-domain conformance result is therefore preserved with its exact environment provenance; it is not a latency or dependency-conformance claim.

## Evidence Boundary

The established statement is:

$$
Feasible(TTA_{restricted})
=
Feasible(TTA_{baseline}) \cap \Gamma(x)
$$

for the frozen finite v2 staging-placement and adjacent-path space. It does not establish performance benefit, general tensor realization, or Gate 1B candidate quality.
