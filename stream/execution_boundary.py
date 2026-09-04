"""Lightweight runtime evidence for code paths that must remain prepare-only."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum


class ExecutionEvent(str, Enum):
    """Operations whose absence is part of the Gate 2A evidence boundary."""

    TTA_CONSTRUCT = "tta_construct"
    TTA_SOLVE = "tta_solve"
    STRUCTURAL_EXHAUSTIVE = "structural_exhaustive"
    STRUCTURAL_VARIABLE_ELIMINATION = "structural_variable_elimination"


class ForbiddenExecutionError(RuntimeError):
    """Raised when an observed operation violates an active prepare-only boundary."""


@dataclass(slots=True)
class ExecutionAudit:
    """Per-context event counters with optional fail-closed forbidden events."""

    forbidden: frozenset[ExecutionEvent]
    counts: dict[ExecutionEvent, int] = field(default_factory=lambda: {event: 0 for event in ExecutionEvent})

    def record(self, event: ExecutionEvent) -> None:
        self.counts[event] += 1
        if event in self.forbidden:
            raise ForbiddenExecutionError(f"forbidden execution event: {event.value}")

    def manifest(self) -> dict[str, int]:
        return {event.value: self.counts[event] for event in ExecutionEvent}


_ACTIVE_AUDIT: ContextVar[ExecutionAudit | None] = ContextVar("stream_execution_audit", default=None)


@contextmanager
def audit_execution(*, forbidden: frozenset[ExecutionEvent] = frozenset()) -> Iterator[ExecutionAudit]:
    """Observe execution events in the current context without affecting ordinary runs."""

    audit = ExecutionAudit(forbidden)
    token = _ACTIVE_AUDIT.set(audit)
    try:
        yield audit
    finally:
        _ACTIVE_AUDIT.reset(token)


def record_execution_event(event: ExecutionEvent) -> None:
    """Record an event when an audit is active; otherwise remain a no-op."""

    audit = _ACTIVE_AUDIT.get()
    if audit is not None:
        audit.record(event)


__all__ = [
    "ExecutionAudit",
    "ExecutionEvent",
    "ForbiddenExecutionError",
    "audit_execution",
    "record_execution_event",
]
