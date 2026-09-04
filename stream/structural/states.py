"""Finite v0 operator, tensor-realization, and distribution templates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from stream.workload.tensor_domain import AffineBox, TensorFragment


class MaterializationMode(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    STREAMING = "streaming"


class DistributionKind(StrEnum):
    LOCAL = "local"
    BLOCK = "block"
    REPLICATED = "replicated"
    SHARED = "shared"


class DistributionPlanKind(StrEnum):
    LOCAL = "local"
    BLOCK = "block"
    REPLICATE = "replicate"
    SHARED = "shared"
    UNICAST_FANOUT = "unicast_fanout"
    MULTICAST_TREE = "multicast_tree"
    FULL_MATERIALIZE = "full_materialize"


@dataclass(frozen=True, slots=True)
class DistributionTemplate:
    name: str
    kind: DistributionKind
    zones: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.zones:
            raise ValueError("distribution templates require a name and at least one zone")


@dataclass(frozen=True, slots=True)
class OperatorState:
    name: str
    iteration_tile: AffineBox
    output_layout: tuple[int, ...]
    zone: str
    integer_cost: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.zone:
            raise ValueError("operator states require a name and hardware zone")
        if not self.output_layout or sorted(self.output_layout) != list(range(len(self.output_layout))):
            raise ValueError("output_layout must be a non-empty axis permutation")
        if self.integer_cost < 0:
            raise ValueError("integer_cost cannot be negative")


@dataclass(frozen=True, slots=True)
class TensorRealization:
    name: str
    fragments: tuple[TensorFragment, ...]
    distribution: DistributionTemplate
    materialization: MaterializationMode

    def __post_init__(self) -> None:
        if not self.name or not self.fragments:
            raise ValueError("tensor realizations require a name and at least one fragment")
        rank = self.fragments[0].domain.rank
        if any(fragment.domain.rank != rank for fragment in self.fragments):
            raise ValueError("all fragments in one realization must have the same rank")


@dataclass(frozen=True, slots=True)
class DistributionPlan:
    name: str
    kind: DistributionPlanKind
    consumers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ValueError("distribution plans require a name and kind")
        object.__setattr__(self, "kind", DistributionPlanKind(self.kind))
        if len(set(self.consumers)) != len(self.consumers):
            raise ValueError("distribution-plan consumers must be unique")
