# Tensor-Realization Gate 1A Contract

Gate 1A asks whether every preregistered structural literal can be compiled into the existing STREAM pipeline without semantic relaxation or unintended restriction. It does not optimize latency or change TETRA.

## Exactness Definition

For structural assignment $x$, let $C(x)$ be the compiled problem and `ReferencePipeline(x)` the ordinary STREAM pipeline with only the restrictions represented by $x$. A literal is `EXACT` only when the two are semantically equivalent:

```text
soundness:    feasible(C(x)) is a subset of the intended concretization
completeness: intended feasible solutions are a subset of feasible(C(x))
```

Unsupported capability is reported with an enum reason. Gate 1A permits only `EXACT` and `UNSUPPORTED`; it has no approximate or best-effort mode. An assignment is exact only if every literal is exact.

## Two-Stage Compiler

The compiler follows the production scheduler order:

```text
StructuralMappingContract
  -> pre-transfer filters: operator tiling and compute-core groups
  -> build_transfer_graph()
  -> update_mapping()
  -> post-transfer filters: supported tensor/path/reuse restrictions
  -> update_cost_lut()
  -> generate_ssis()
  -> get_timeslots(filtered_mapping)
  -> unchanged TTA
```

Post-transfer path filtering must precede timeslot construction because timeslot feasibility reads each path's `links_used`. Current literals that cannot be represented exactly—including pipeline materialization, independent tensor placement, output layout, and exact reuse—remain explicitly unsupported.

## Conformance Evidence

Each exact literal requires three evidence layers:

1. Candidate-set equality: compiled options equal reference options intersected with the allowed set.
2. Violation-witness exclusion: $C(x)\land\neg l$ has no feasible semantic solution.
3. Bounded completeness: compiled and reference feasible sets are identical after canonical semantic projection.

Baseline round trips compare a canonical semantic SHA-256 manifest containing workload, accelerator, cost LUT, candidate sets, timeslots, reuse domains, constraint families, and evaluation configuration. Raw backend hashes are diagnostic only. Repeated compilation must produce the same semantic hash.

The frozen denominator is `stream/structural/contracts/gate1a_intended_space_v1.json`. It fixes at least 1,000 deterministic assignments, three compilations per assignment, and the complete materialization-by-distribution-by-DAG coverage matrix.

## Verdicts

- `PASS`: intended downstream coverage is 100%, with zero semantic, round-trip, or determinism violations.
- `NARROW`: exact coverage is incomplete, but every supported literal is proven exact, every other literal is explicit, the baseline round trip passes, and a preregistered nontrivial subset remains.
- `FAIL`: any false `EXACT`, silent relaxation, unintended restriction, baseline mismatch, or nondeterminism occurs.
- `NOT_RUN`: any required evidence layer or sample count is incomplete.

Missing capability is therefore a narrowing result; incorrect semantics is a failure.
