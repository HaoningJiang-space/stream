"""Derived operand-access queries (relevancy, tile footprints, dependency regions) over a node's xDSL affine operand
maps."""

from __future__ import annotations

from collections.abc import Mapping

from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)

from stream.workload.affine_footprint import Interval, footprint
from stream.workload.node import HasIterationSpace
from stream.workload.steady_state.iteration_space import LoopEffect
from stream.workload.tensor import Tensor

__all__ = [
    "Interval",
    "map_dim_positions",
    "relevancy",
    "operand_relevancy",
    "footprint",
    "compose_dependency",
]


def _position(dim: int | AffineDimExpr) -> int:
    if isinstance(dim, AffineDimExpr):
        return dim.position
    return int(dim)


def _dims_in_expr(expr: AffineExpr) -> frozenset[int]:
    """Positions of every dimension that appears with a non-zero coefficient in ``expr``."""
    if isinstance(expr, AffineDimExpr):
        return frozenset({expr.position})
    if isinstance(expr, AffineBinaryOpExpr):
        return _dims_in_expr(expr.lhs) | _dims_in_expr(expr.rhs)
    return frozenset()


def map_dim_positions(affine_map: AffineMap) -> frozenset[int]:
    """Positions of all iteration dimensions that index any result of ``affine_map``."""
    positions: frozenset[int] = frozenset()
    for result in affine_map.results:
        positions |= _dims_in_expr(result)
    return positions


def relevancy(node: HasIterationSpace, operand: Tensor, dim: int | AffineDimExpr) -> LoopEffect:
    """How iteration dimension ``dim`` affects ``operand``: VARYING if it indexes the operand, INVARIANT if a node dim
    that does not, ABSENT if not a node dim."""
    pos = _position(dim)
    if pos < 0 or pos >= node.num_dims:
        return LoopEffect.ABSENT
    if pos in map_dim_positions(node.get_mapping(operand)):
        return LoopEffect.VARYING
    return LoopEffect.INVARIANT


def operand_relevancy(node: HasIterationSpace, operand: Tensor) -> dict[int, LoopEffect]:
    """Relevancy of every node iteration dimension for ``operand``, keyed by dimension position."""
    varying = map_dim_positions(node.get_mapping(operand))
    return {pos: (LoopEffect.VARYING if pos in varying else LoopEffect.INVARIANT) for pos in range(node.num_dims)}


def compose_dependency(
    producer_out: AffineMap,
    consumer_in: AffineMap,
    consumer_tile: Mapping[int | AffineDimExpr, range],
) -> dict[int, range]:
    """Producer iteration region (``range`` per producer dim) needed for the shared-tensor slice a consumer tile reads;
    producer dims not indexing the shared tensor are omitted."""
    tensor_slice = footprint(consumer_in, consumer_tile)
    if len(tensor_slice) != len(producer_out.results):
        raise ValueError(
            f"shared tensor rank mismatch: producer produces {len(producer_out.results)} indices, "
            f"consumer reads {len(tensor_slice)}"
        )
    producer_region: dict[int, range] = {}
    for result, index_range in zip(producer_out.results, tensor_slice, strict=True):
        if not isinstance(result, AffineDimExpr):
            raise NotImplementedError(
                "compose_dependency requires a single-dimension (permutation) producer output map; "
                "got a composite expression. Use the islpy exact path."
            )
        producer_region[result.position] = index_range
    return producer_region
