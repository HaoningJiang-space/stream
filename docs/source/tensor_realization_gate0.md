# Tensor-Realization Gate 0 Contract

Gate 0 asks one bounded question: given finite operator, tensor-realization, and distribution-plan template libraries, can an integer structural objective be factorized and optimized by variable elimination with exactly the same result as exhaustive enumeration?

## Scope

The v0 domain is limited to dense, static affine integer boxes. It supports contiguous fragmentation, layout permutations, coarse hardware zones, full or streaming materialization, fork/join graphs, and integer movement costs. Sparse or dynamic tensors, irregular gather/scatter, compression, scheduling, latency estimation, and inter-core reduction-axis splits are out of scope. TETRA is not modified or invoked.

The finite variables are:

- `OperatorState`: iteration tile, produced-tensor layout, and hardware zone;
- `TensorRealization`: fragments, distribution template, and materialization mode;
- `DistributionPlan`: one plan covering all consumers of a tensor.

## Accounting Invariant

Every modeled physical event has one stable key and exactly one owning factor. Distribution factors own shared or non-additive events such as materialization. Consumer factors own only terminal reads and retiles. Duplicate event keys make an assignment invalid rather than silently double-charging it.

`direct_cost.py` provides independent full-assignment semantics. For every enumerated assignment, tests require identical legality and event ledgers:

```text
direct events == factorized events
```

This check prevents exhaustive enumeration and variable elimination from agreeing on the same incorrect factorization.

## Exactness Gate

All costs are non-negative integers. Missing factor-table entries denote illegal assignments. Gate 0 permits no beam search, A*, dominance, Pareto truncation, floating-point tolerance, or approximate pruning. State reduction comes only from legality checks and exact min-sum elimination.

The required conditions are:

```text
OPT(VE) == OPT(direct exhaustive enumeration)
canonical(VE assignment) belongs to canonical(exhaustive argmin)
```

The fixed micro-suite covers an edge, a chain, GEMM-to-GEMM, Conv-to-Conv, a residual diamond, and fork-to-join. It includes explicit shared-cost and layout-mismatch cases. “Exact” refers only to this finite v0 structural model, not physical latency, unrestricted hardware mapping, or the STREAM/TETRA optimum.
