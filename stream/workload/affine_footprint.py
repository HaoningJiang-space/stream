"""Lightweight affine-map footprint queries over dense iteration boxes."""

from __future__ import annotations

from collections.abc import Mapping

from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)

Interval = tuple[int, int]
"""Inclusive ``(low, high)`` bounds of an integer quantity."""


def _position(dim: int | AffineDimExpr) -> int:
    return dim.position if isinstance(dim, AffineDimExpr) else int(dim)


def _constant(expr: AffineExpr) -> int | None:
    return expr.value if isinstance(expr, AffineConstantExpr) else None


def _range_bounds(extent: range) -> Interval:
    if len(extent) == 0:
        raise ValueError("tile range must be non-empty")
    return extent[0], extent[-1]


def _interval_of_expr(expr: AffineExpr, box: Mapping[int, Interval]) -> Interval:
    """Exact inclusive interval of an affine expression over an iteration box."""
    if isinstance(expr, AffineConstantExpr):
        return expr.value, expr.value
    if isinstance(expr, AffineDimExpr):
        if expr.position not in box:
            raise ValueError(f"tile does not bound dimension d{expr.position}")
        return box[expr.position]
    if isinstance(expr, AffineBinaryOpExpr):
        if expr.kind == AffineBinaryOpKind.Add:
            low_l, high_l = _interval_of_expr(expr.lhs, box)
            low_r, high_r = _interval_of_expr(expr.rhs, box)
            return low_l + low_r, high_l + high_r
        if expr.kind == AffineBinaryOpKind.Mul:
            coefficient = _constant(expr.lhs)
            other = expr.rhs
            if coefficient is None:
                coefficient = _constant(expr.rhs)
                other = expr.lhs
            if coefficient is None:
                raise NotImplementedError("product of two non-constant affine expressions is not box-representable")
            low, high = _interval_of_expr(other, box)
            return (
                (coefficient * low, coefficient * high) if coefficient >= 0 else (coefficient * high, coefficient * low)
            )
        raise NotImplementedError(f"affine operator {expr.kind} has no exact box interval; use the islpy exact path")
    raise NotImplementedError(f"unsupported affine expression {type(expr).__name__}")


def footprint(affine_map: AffineMap, tile: Mapping[int | AffineDimExpr, range]) -> tuple[range, ...]:
    """Return one exact contiguous result range per affine-map result."""
    box = {_position(dim): _range_bounds(extent) for dim, extent in tile.items()}
    return tuple(
        range(low, high + 1) for low, high in (_interval_of_expr(result, box) for result in affine_map.results)
    )
