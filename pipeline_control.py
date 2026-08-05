"""Shared cooperative-cancellation primitives for the local pipeline."""

from __future__ import annotations

from typing import Callable, Optional


CancelCheck = Callable[[], bool]


class PipelineCancelledError(RuntimeError):
    """Raised at a safe checkpoint when cooperative cancellation is requested."""


def raise_if_cancelled(
    cancel_check: Optional[CancelCheck],
    *,
    phase: str | None = None,
) -> None:
    """Raise the dedicated stop signal when an optional callback requests it."""

    if cancel_check is not None and cancel_check():
        detail = f" during {phase}" if phase else ""
        raise PipelineCancelledError(f"pipeline cancellation requested{detail}")


__all__ = ["CancelCheck", "PipelineCancelledError", "raise_if_cancelled"]
