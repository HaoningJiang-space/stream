# Real-Workload Lifting Validity: Gate 2A

Gate 2A asks whether the production STREAM frontend can deterministically lift real workloads into a valid, auditable Gate 1A-v2 structural problem. It is a prepare-only validity gate, not a variable-elimination benchmark or latency experiment.

## Frozen Denominator

The workload-selection policy was fixed before lifting repairs and covers four topology families: Llama/SwiGLU FFN, FSRCNN, ResNet18, and an attention head. All use the Ironwood hardware description and generic production mapping. Exact paths and input hashes are recorded in the machine-readable contract and result artifact. An accepted run must execute from a clean Git checkout whose HEAD matches the recorded source commit; an unversioned synchronized snapshot cannot produce `PASS`.

## Preparation Boundary

For every workload, the harness runs the production accelerator and ONNX parsers, normalization expansion, generic fusion/mapping generation, per-group mapping parsing, kernel-state lifting, tiling generation, transfer-graph construction, mapping update, SSIS generation, multiplicity calculation, and resource-aware timeslot construction. It then extracts placement and path domains directly from the prepared mapping.

The core-cost LUT contains deterministic unit entries because Gate 2A never consumes cost values. A runtime execution boundary instruments TTA construction/solve and the exhaustive/VE entry points; any such event aborts the attempt, while successful manifests serialize zero counters. The harness therefore does not construct or solve TTA and does not run structural optimization, VE, candidate ranking, latency evaluation, or Gate 2B/C/D.

## Validity Conditions

A workload is valid only if every fusion group has legal affine tensor domains, complete producer/consumer identity, non-empty placement/path domains, complete tensor provenance, no unsupported mapping fallback, and a complete preparation trace. Every supplied ONNX operand omitted by the current semantic parser must be declared by the parser, match a frozen contract rule, and appear in the manifest with node, input index, tensor, shape, source, and reason. This makes existing limitations such as unmodeled Conv bias operands explicit rather than silent.

Fresh preparation is repeated twice in independent Python processes with distinct hash seeds, and the canonical semantic manifests must have identical SHA-256 hashes. The verdict enforces the frozen Python and OR-Tools minimum versions, records the contract hash and every criterion, compares clean Git/source snapshots before and after preparation, and requires the report destination to be outside the checkout.

The overall verdict is

$$
\mathrm{PASS}\iff
\forall W\in\mathcal W_{frozen},\ \mathrm{Prepare}(W)=\mathrm{VALID}.
$$

Any failure yields `NOT_RUN`, not an opportunity or scalability failure. Stable reason codes distinguish frontend, affine-domain, constant-tensor, transfer-domain, shared-input, identity, empty-domain, fallback, and nondeterminism failures. Workload-specific fallback and forced dimension merging are prohibited.

## Claim Boundary

Gate 2A PASS establishes production-path lifting validity only. It does not show that the v2 space is non-degenerate, coupled, worth searching, or predictive on real networks. Those questions remain frozen behind Gate 2B, Gate 2C, and Gate 2D.
