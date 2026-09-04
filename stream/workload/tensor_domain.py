"""Dense affine-box tensor fragments for structural mapping.

The v0 contract deliberately models dense, static, integer index boxes only.
Sparse/data-dependent accesses, dynamic shapes, compression, and reduction-axis
partitioning are outside this module's scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xdsl.ir.affine import AffineMap


@dataclass(frozen=True, slots=True)
class AffineBox:
    """Half-open dense integer box ``lower <= index < upper``."""

    lower: tuple[int, ...]
    upper: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.lower or len(self.lower) != len(self.upper):
            raise ValueError("AffineBox bounds must have the same non-zero rank")
        if any(lo < 0 or hi <= lo for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("AffineBox requires 0 <= lower < upper on every axis")

    @property
    def rank(self) -> int:
        return len(self.lower)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(hi - lo for lo, hi in zip(self.lower, self.upper, strict=True))

    @property
    def elements(self) -> int:
        return prod(self.shape)

    def intersection(self, other: AffineBox) -> AffineBox | None:
        if self.rank != other.rank:
            raise ValueError("cannot intersect boxes of different rank")
        lower = tuple(max(a, b) for a, b in zip(self.lower, other.lower, strict=True))
        upper = tuple(min(a, b) for a, b in zip(self.upper, other.upper, strict=True))
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            return None
        return AffineBox(lower, upper)


@dataclass(frozen=True, slots=True)
class TensorFragment:
    """A logical tensor fragment and its physical axis order."""

    domain: AffineBox
    layout: tuple[int, ...]
    role: str = "normal"

    def __post_init__(self) -> None:
        if sorted(self.layout) != list(range(self.domain.rank)):
            raise ValueError("layout must be a permutation of fragment axes")
        if self.role not in {"normal", "halo", "broadcast", "padding"}:
            raise ValueError(f"unsupported v0 fragment role: {self.role}")


@dataclass(frozen=True, slots=True)
class TensorTileDomain:
    """Static dense tensor domain with an integer byte width."""

    shape: tuple[int, ...]
    dtype_bytes: int

    def __post_init__(self) -> None:
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("tensor shape must contain positive static dimensions")
        if self.dtype_bytes <= 0:
            raise ValueError("dtype_bytes must be positive")

    @property
    def full_box(self) -> AffineBox:
        return AffineBox((0,) * len(self.shape), self.shape)

    def partition(self, axis: int, parts: int, *, layout: tuple[int, ...] | None = None) -> tuple[TensorFragment, ...]:
        """Partition one axis into balanced contiguous fragments."""
        if not 0 <= axis < len(self.shape):
            raise ValueError(f"axis {axis} is outside tensor rank {len(self.shape)}")
        if parts <= 0 or parts > self.shape[axis]:
            raise ValueError("parts must be positive and no larger than the partitioned extent")
        axis_order = layout or tuple(range(len(self.shape)))
        base, extra = divmod(self.shape[axis], parts)
        fragments: list[TensorFragment] = []
        start = 0
        for part in range(parts):
            stop = start + base + (1 if part < extra else 0)
            lower = [0] * len(self.shape)
            upper = list(self.shape)
            lower[axis], upper[axis] = start, stop
            fragments.append(TensorFragment(AffineBox(tuple(lower), tuple(upper)), axis_order))
            start = stop
        return tuple(fragments)

    def demand_from_affine(
        self,
        affine_map: AffineMap,
        iteration_tile: dict[int, range],
        *,
        layout: tuple[int, ...] | None = None,
        role: str = "normal",
    ) -> TensorFragment:
        """Map a dense iteration box to its clipped operand-demand box."""
        from stream.workload.affine_footprint import footprint  # noqa: PLC0415

        ranges = footprint(affine_map, iteration_tile)
        if len(ranges) != len(self.shape):
            raise ValueError("affine result rank does not match the tensor rank")
        lower = tuple(max(0, r.start) for r in ranges)
        upper = tuple(min(size, r.stop) for size, r in zip(self.shape, ranges, strict=True))
        if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("affine demand does not intersect the logical tensor domain")
        return TensorFragment(
            AffineBox(lower, upper),
            layout or tuple(range(len(self.shape))),
            role,
        )


def fragment_overlap_bytes(producer: TensorFragment, consumer: TensorFragment, dtype_bytes: int) -> int:
    """Exact dense bytes in the logical intersection of two fragments."""
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")
    overlap = producer.domain.intersection(consumer.domain)
    return 0 if overlap is None else overlap.elements * dtype_bytes


def validate_no_reduction_split(partition_axes: tuple[int, ...], reduction_axes: tuple[int, ...]) -> None:
    """Enforce the v0 boundary: inter-core partitions cannot split reductions."""
    overlap = set(partition_axes) & set(reduction_axes)
    if overlap:
        raise ValueError(f"v0 forbids reduction-axis inter-core splits: {sorted(overlap)}")
