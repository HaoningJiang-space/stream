"""Canonical keys for comparing symmetry-equivalent structural assignments."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from typing import Any


def _default_value_key(value: Any) -> Hashable:
    key = getattr(value, "canonical_key", None)
    if callable(key):
        return key()
    return value


def canonical_assignment(
    assignment: Mapping[str, Any],
    value_key: Callable[[Any], Hashable] = _default_value_key,
) -> tuple[tuple[str, Hashable], ...]:
    return tuple((name, value_key(value)) for name, value in sorted(assignment.items()))
