# Gate 1A-v4: Shared-Tensor Operator-Template Coupling

## Verdict

Gate 1A-v4 is **PASS / COUPLED_LEGALITY**. The accepted run exhaustively crossed seven atomic
`(core group, split template)` choices for both consumers of a shared tensor in residual and fork-join micro-DAGs.
It proves a non-Cartesian binary legality relation in the finite pre-TTA structural model.

The accepted run used commit `a395f86286b4cd7062292fac4732ff3741c59ca0` on `eex004`, Python 3.12.11,
OR-Tools 9.15.6755, xDSL 0.29.1, ZigZag 3.8.5, and STREAM 1.14.1. The complete evidence bundle is stored in
`artifacts/gate1a-v4/`.

| Check | Result |
|---|---:|
| Frozen joint assignments | 98 |
| Exactly prepared through timeslots | 90 |
| Registered unsupported assignments | 8 |
| Compiler/direct-reference equality | 98 / 98 |
| Compatibility prediction equality | 98 / 98 |
| Single-change failures restored by joint changes | 8 |
| Baseline round trips | 2 / 2 |
| Serial versus 8-worker replay | Exact |
| Silent relaxations / unexpected failures | 0 / 0 |

The report SHA-256 is `f30b6f07c0649ccf5d2666465c9e23bca730098d2da218c43972da93ca4be396`; the portable
run-manifest SHA-256 is `41dc9f69451ab8971c08da03d8e21e2feacd39e075aa860d51f71ade9e08abb7`.
The runtime image SHA-256 was `04495407bc82a64adedf482d4a734d04cfd992651e301055b66a6aef885aaafa`.

## Coupling Witness

Let template `4` use `D0=2` and template `6` use `D0=4`. For both frozen shared tensors, the observed relation is

\[
R(4,4)=R(6,6)=1,\qquad R(4,6)=R(6,4)=0.
\]

If legality were separable into independent unary domains, the two diagonal feasible assignments would force both
off-diagonal assignments to be feasible. The forbidden cross therefore proves a genuine binary compatibility factor.
The factor is a legality constraint, not yet a cross-operator performance cost.

## Optimization Opportunity and Boundary

Gate 2E-A previously found 5,873 **potential** local templates over 70 real-workload operators and 57 potential
compute-to-compute coupling edges. Its largest per-group nominal spaces range from \(10^{8.32}\) to \(10^{18.01}\).
Gate 1A-v4 now proves the mechanism that prevents independent per-operator argmin: shared-tensor consumers must choose
compatible tensor-relevant splits, and coordinated changes can recover choices rejected in isolation.

This Gate does not prove that all 5,873 states lift on real workloads, that the performance objective is coupled, or
that latency improves. The next Gate must lift the exact compatibility relation to frozen real workloads, report the
legal joint state count and factor width, and only then decide between direct factorized optimization and VE.
