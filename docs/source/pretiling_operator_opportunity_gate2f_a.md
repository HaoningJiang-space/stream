# Gate 2F-A: Pre-Tiling Operator-Template Opportunity

## Question

Gate 2E-A counted templates from Gate 2A manifests after baseline tiling. Gate 2F-A preserves that historical result but asks the corrected question:

> At the production seam immediately after `MappingParser`, how large is the finite paired `(core group, split)` space, and which shared tensors induce non-separable compatibility relations?

The frozen workload denominator remains SwiGLU, FSRCNN, ResNet18, and an attention head. Each preparation stops before `KernelState`, `TilingGeneration`, scheduler construction, and TTA.

The input audit binds every ONNX and hardware artifact byte-for-byte to the accepted Gate 2A evidence and requires equality of Gate 2A's recorded Python/package manifest. A separate hash-pinned reference was captured from Gate 2A's clean source commit at this exact MappingParser seam. The current graph topology, tensor identities/shapes/types, operation and fusion metadata, loop extents, affine operand maps, core, local-split, fused-group, runtime-argument, and generated mapping-file manifests must equal that reference. This directly compares the consumed workload semantics and compiler outputs instead of relying on a hand-maintained transitive source-code allowlist. Gate 2A's later post-`update_mapping()` global `z` names are not compared with MappingParser's local `D` names because they belong to different compiler seams.

## Template domain

For a parsed core pool of width (W), output-parallel split factors must divide the pre-tiling loop extents and have product (k\mid W). Core order is the parsed mapping order. Width-(k) placements are aligned chunks:

\[
P[j:j+k],\qquad j=0,k,\ldots,W-k.
\]

The parsed baseline is accepted only if its core pool and tiling already form one valid paired template. Every generated state is passed back through the existing compiler legality path.

## Shared-tensor signature proxy

For each tensor with at least two computation consumers, a consumer template is projected onto its selected dimensions that index that tensor. Let consumer (i) have (N_i) states, (e_i) empty-signature states, and (m_i(s)) states with non-empty signature (s). The local compatible-tuple count is

\[
C=E+\sum_s\left[\prod_i(e_i+m_i(s))-E\right],
\qquad E=\prod_i e_i.
\]

A relation is non-Cartesian only when (C) is smaller than the product of its unary support projections. The analytical count is independently checked by direct enumeration of signature classes.

## Evidence boundary

This is exploratory, preflight-informed, prepare-only evidence. The signature rule is a pre-tiling proxy whose abstract empty-or-equal form passed the Gate 1A-v4 micro test. Gate 2F-A does **not** establish real-workload post-tiling equivalence, TTA executability, global feasible-space size, factor independence, VE scalability, or latency improvement.

If the space is large but the factor graph is sparse, the next search should exploit connected components and unary local choices rather than apply monolithic exhaustive enumeration.
