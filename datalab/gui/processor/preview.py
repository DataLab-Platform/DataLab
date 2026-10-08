"""Isolated, bounded speculative computation without workspace side effects."""

from __future__ import annotations

import copy
import multiprocessing
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

from guidata.dataset import DataSet
from qtpy import QtCore as QC
from sigima.config import options as sigima_options
from sigima.objects import ImageObj, SignalObj

from datalab.gui.processor.base import run_with_env
from datalab.gui.processor.catcher import CompOut

__all__ = ["PreviewController", "PreviewExecutor", "PreviewExecutorCache"]


class PreviewExecutor:
    """Own a private process whose startup and disposal never block Qt."""

    def __init__(self) -> None:
        self._threads = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview")
        self._stop = threading.Event()
        self._pool = None
        self._closed = False

    def submit(self, function: Callable, args: tuple) -> Future:
        """Submit independent arguments and the current scientific configuration."""
        if self._closed:
            raise RuntimeError("Preview executor is closed")
        return self._threads.submit(
            self._execute, function, args, sigima_options.get_env()
        )

    def _execute(self, function: Callable, args: tuple, environment: str) -> CompOut:
        try:
            if self._stop.is_set():
                return CompOut(cancelled=True)
            if self._pool is None:
                self._pool = multiprocessing.get_context("spawn").Pool(1)
            if self._stop.is_set():
                return CompOut(cancelled=True)
            result = self._pool.apply_async(run_with_env, (function, args, environment))
            while not result.ready():
                if self._stop.wait(0.02):
                    return CompOut(cancelled=True)
            return result.get()
        except Exception:
            return CompOut(error_msg=traceback.format_exc())
        finally:
            if self._stop.is_set():
                self._dispose()

    def _dispose(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def close(self, wait: bool = False) -> None:
        """Cancel work and release resources; waiting is intended for tests only."""
        if not self._closed:
            self._closed = True
            self._stop.set()
            self._threads.submit(self._dispose)
        self._threads.shutdown(wait=wait)


class PreviewExecutorCache:
    """Retain at most one idle preview executor between dialogs."""

    def __init__(self, executor_factory: Callable = PreviewExecutor) -> None:
        self._executor_factory = executor_factory
        self._idle = None
        self._leased = {}
        self._generation = 0
        self._closed = False

    def acquire(self) -> PreviewExecutor:
        """Return an executor owned exclusively by the caller."""
        if self._closed:
            raise RuntimeError("Preview executor cache is closed")
        executor = self._idle
        if executor is None:
            executor = self._executor_factory()
        else:
            self._idle = None
        self._leased[executor] = self._generation
        return executor

    def release(self, executor: PreviewExecutor, reusable: bool = True) -> None:
        """Return an executor to the idle slot or close it."""
        if executor not in self._leased:
            return
        generation = self._leased.pop(executor)
        if (
            self._closed
            or not reusable
            or generation != self._generation
            or self._idle is not None
        ):
            executor.close()
        else:
            self._idle = executor

    def reset(self) -> None:
        """Discard idle state and reject executors from the previous generation."""
        self._generation += 1
        if self._idle is not None:
            self._idle.close()
            self._idle = None

    def close(self) -> None:
        """Close every owned executor and reject future acquisitions."""
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        executors = list(self._leased)
        self._leased.clear()
        if self._idle is not None:
            executors.append(self._idle)
            self._idle = None
        for executor in executors:
            executor.close()


class PreviewController(QC.QObject):
    """Keep one active computation and only the latest pending snapshot.

    The owner schedules requests after validating its form. ``mark_dirty``
    prevents older parameters from being presented as current. ``invalidate``
    additionally discards results from an obsolete source or invalid form.
    """

    SIG_RESULT = QC.Signal(object, bool)
    SIG_ERROR = QC.Signal(str)
    SIG_BUSY = QC.Signal(bool)

    def __init__(
        self,
        function: Callable,
        parent: QC.QObject | None = None,
        executor_factory: Callable = PreviewExecutor,
        executor_cache: PreviewExecutorCache | None = None,
    ) -> None:
        super().__init__(parent)
        self.function = function
        self._executor_factory = executor_factory
        self._executor_cache = executor_cache
        self._executor = None
        self._future = None
        self._pending = None
        self._active_key = None
        self._active_source = None
        self._current_result = None
        self._epoch = 0
        self._revision = 0
        self._enabled = False
        self._timer = QC.QTimer(self)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self.poll)

    def set_enabled(self, enabled: bool) -> None:
        """Enable requests, or cancel all speculative work."""
        if enabled:
            self._enabled = True
        else:
            self.close()

    def mark_dirty(self) -> None:
        """Forget pending parameters without publishing older ones as current."""
        self._revision += 1
        self._pending = None
        self._current_result = None

    def invalidate(self) -> None:
        """Invalidate even provisional results after a source/validity change."""
        self.mark_dirty()
        self._epoch += 1

    def request(self, source: SignalObj | ImageObj, param: DataSet) -> None:
        """Snapshot validated data on the GUI thread and enqueue its computation."""
        if not self._enabled:
            return
        self._revision += 1
        try:
            args = copy.deepcopy((source, param))
        except Exception:
            self.invalidate()
            self.SIG_ERROR.emit(traceback.format_exc())
            return
        self._pending = ((self._epoch, self._revision), args, source)
        self._current_result = None
        if self._future is None:
            self._start_pending()

    def _start_pending(self) -> None:
        if self._pending is None or not self._enabled:
            return
        self._active_key, args, self._active_source = self._pending
        self._pending = None
        try:
            if self._executor is None:
                if self._executor_cache is None:
                    self._executor = self._executor_factory()
                else:
                    self._executor = self._executor_cache.acquire()
            self._future = self._executor.submit(self.function, args)
        except Exception:
            self._future = None
            self._release_executor(reusable=False)
            self.SIG_ERROR.emit(traceback.format_exc())
            return
        self._timer.start()
        self.SIG_BUSY.emit(True)

    def poll(self) -> None:
        """Consume a completed future without waiting or pumping Qt events."""
        if self._future is None or not self._future.done():
            return
        future, key = self._future, self._active_key
        self._future = None
        self._timer.stop()
        try:
            output = future.result()
        except Exception:
            output = CompOut(error_msg=traceback.format_exc())
        if self._enabled and key[0] == self._epoch:
            current = key[1] == self._revision
            if output.error_msg and current:
                self.SIG_ERROR.emit(output.error_msg)
            elif not output.cancelled and output.result is not None:
                if current:
                    self._current_result = (
                        key,
                        self._active_source,
                        output,
                    )
                self.SIG_RESULT.emit(output, current)
        if self._pending is not None:
            self._start_pending()
        else:
            self.SIG_BUSY.emit(False)

    def take_current_result(self, source: SignalObj | ImageObj) -> CompOut | None:
        """Detach the completed result when it still matches *source*."""
        candidate = self._current_result
        self._current_result = None
        if candidate is None or not self._enabled:
            return None
        key, candidate_source, output = candidate
        if key != (self._epoch, self._revision) or candidate_source is not source:
            return None
        return output

    def close(self) -> None:
        """Disconnect the view from outstanding work and cancel privately."""
        reusable = self._future is None or self._future.done()
        self._enabled = False
        self.invalidate()
        self._timer.stop()
        self._future = None
        self._active_source = None
        self._current_result = None
        self._release_executor(reusable=reusable)
        self.SIG_BUSY.emit(False)

    def _release_executor(self, reusable: bool) -> None:
        """Release the current executor according to its completion state."""
        executor = self._executor
        self._executor = None
        if executor is None:
            return
        if self._executor_cache is None:
            executor.close()
        else:
            self._executor_cache.release(executor, reusable=reusable)
