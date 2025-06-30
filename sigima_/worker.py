# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Worker (:mod:`sigima_.worker`)
------------------------------

This module provides a minimal interface for executing long-running operations
with support for progress reporting and cancellation.

It defines a generic protocol (`CallbackWorkerProtocol`) and a console-based
implementation (`TextCallbackWorker`), suitable for non-GUI environments.

Example:

.. code-block:: python

    from sigima_.worker import TextCallbackWorker

    def long_task(worker=None):
        for i in range(10):
            if worker and worker.was_canceled():
                return None
            worker.set_progress(i / 10)
        return "done"

    worker = TextCallbackWorker()
    result = long_task(worker=worker)
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class CallbackWorkerProtocol(Protocol):
    """Protocol defining the minimal interface for progress/cancel-aware workers."""

    def set_progress(self, value: float) -> None:
        """Update progress (between 0.0 and 1.0)."""

    def was_canceled(self) -> bool:
        """Check whether the operation has been canceled."""


class TextCallbackWorker:
    """Minimal worker implementation for console-based environments.

    Provides `set_progress()` and `was_canceled()` methods for use in long-running
    computations. Intended for use in `sigima_` where no GUI (e.g. Qt) is available.

    Attributes:
        _canceled: Whether the operation was canceled.
        _progress: Most recent progress value.
    """

    def __init__(self) -> None:
        self._canceled = False
        self._progress = 0.0

    def set_progress(self, value: float) -> None:
        """Set the progress of the operation (prints to console).

        Args:
            value: Float between 0.0 and 1.0.
        """
        self._progress = min(max(value, 0.0), 1.0)
        percent = int(self._progress * 100)
        print(f"[sigima] Progress: {percent}%")

    def was_canceled(self) -> bool:
        """Return whether the operation has been canceled."""
        return self._canceled

    def cancel(self) -> None:
        """Cancel the operation."""
        self._canceled = True

    def get_progress(self) -> float:
        """Return the current progress value."""
        return self._progress
