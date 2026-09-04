# Gate 2D: Distinct Route Opportunity Result

Gate 2D is **FAIL / NO DISTINCT ROUTE OPPORTUNITY**. Increasing the retained route-plan limit from 1 to 4 does not enlarge the route-resource domain on the frozen Ironwood workload set after resource-equivalent plans are removed.

## Matched Experiment

The clean accepted run used public commit `95d40f04a73165f8fedb7cc8f318e3c0b432a137` on eex004 with Python 3.12.11 and OR-Tools 9.15.6755. It performed two process-isolated preparations at each limit for SwiGLU, FSRCNN, ResNet18, and Attention: 16 attempts total, 54.27 seconds wall time, and no TTA construction or solve.

The evidence artifact is `artifacts/gate2d/report.json` with SHA-256 `e2cd18d960ab8983b2ea7c6047374468264d6daf98b9ea410d202af30b361665`.

## Result

All validity and matched-control checks passed:

- clean source and reviewed-diff provenance;
- frozen workload denominator and Gate 2A baseline hashes;
- deterministic lifting with zero forbidden execution events;
- identical route-insensitive structural manifests, tensor/transfer identities, and placement identities;
- unique canonical route-resource signatures; and
- exact inclusion of every limit-1 route resource in the limit-4 domain.

The opportunity requirements failed:

| Workload | Eligible tensors | Distinct path variables at limit 4 | Controlled logical bits |
|---|---:|---:|---:|
| SwiGLU | 5 | 0 | 0.148% |
| FSRCNN | 9 | 0 | 3.155% |
| ResNet18 | 43 | 0 | 3.794% |
| Attention | 8 | 0 | 25.806% |

No staging tensor had strict path-domain growth. The minimum per-workload path-nondegenerate ratio was 0, not the required 1.0.

## Interpretation and Stop Rule

The earlier mechanism probe counted four beam-search outputs per endpoint and appeared to produce 16/80 assignment domains. Independent review showed that count growth did not prove semantic growth. Canonical resource-set comparison revealed that these alternatives were no-op duplicates under the current TTA resource model. They are now removed before plan options reach TTA; the default one-plan behavior remains unchanged.

This result rules out route-plan retention as the missing large search space on the frozen Ironwood model. It does not test latency, route quality on a multi-path topology, or physical fidelity beyond current TTA resource identity.

The next opportunity census must move upstream to finite operator execution templates—inter-core split shape, compute-zone allocation, and the induced tensor endpoints. General VE remains out of scope until that executable space contains measured cross-variable factors.
