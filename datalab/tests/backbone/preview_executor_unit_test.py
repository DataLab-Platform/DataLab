"""Windows-compatible process lifecycle for speculative computations."""

from __future__ import annotations

import multiprocessing
import os
import time

import numpy as np
from sigima.objects import create_signal
from sigima.params import GaussianParam
from sigima.proc.signal import gaussian_filter

from datalab.gui.processor import base
from datalab.gui.processor.preview import PreviewExecutor, PreviewExecutorCache


class FakePreviewExecutor:
    """Record cache lifecycle operations without starting processes."""

    def __init__(self):
        self.close_count = 0

    def close(self):
        """Record resource disposal."""
        self.close_count += 1


class ExecutorFactory:
    """Create and retain fake executors for assertions."""

    def __init__(self):
        self.executors = []

    def __call__(self):
        executor = FakePreviewExecutor()
        self.executors.append(executor)
        return executor


def slow_identity(source, started):
    """Represent an expensive computation that must be cancellable."""
    started.send(True)
    time.sleep(30)
    return source


def get_process_id():
    """Return the spawned worker process identifier."""
    return os.getpid()


def test_cache_reuses_one_idle_executor():
    """An idle executor is reused and never leased to two callers."""
    factory = ExecutorFactory()
    cache = PreviewExecutorCache(factory)

    first = cache.acquire()
    second = cache.acquire()
    assert first is not second
    cache.release(first)
    cache.release(second)
    assert first.close_count == 0
    assert second.close_count == 1

    assert cache.acquire() is first
    cache.release(first)
    cache.release(first)
    assert first.close_count == 0
    cache.close()
    cache.close()
    assert first.close_count == 1


def test_cache_reset_rejects_previous_generation():
    """Reset closes idle state and prevents late returns from repopulating it."""
    factory = ExecutorFactory()
    cache = PreviewExecutorCache(factory)

    leased = cache.acquire()
    cache.reset()
    cache.release(leased)
    assert leased.close_count == 1

    idle = cache.acquire()
    cache.release(idle)
    cache.reset()
    assert idle.close_count == 1
    replacement = cache.acquire()
    assert replacement not in (leased, idle)

    cache.close()
    assert replacement.close_count == 1
    try:
        cache.acquire()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Closed cache accepted an acquisition")


def test_cache_reuses_spawned_process():
    """Completed leases preserve the process across preview sessions."""
    cache = PreviewExecutorCache()
    production_pool = base.POOL
    executor = cache.acquire()
    try:
        first = executor.submit(get_process_id, ()).result(timeout=60)
        assert not first.error_msg, first.error_msg
        cache.release(executor)

        reused = cache.acquire()
        assert reused is executor
        second = reused.submit(get_process_id, ()).result(timeout=60)
        assert not second.error_msg, second.error_msg
        assert second.result == first.result
        cache.release(reused)
        assert base.POOL is production_pool
    finally:
        cache.close()
        executor.close(wait=True)


def test_private_pool_result_and_cleanup():
    """Full-resolution data survive spawn and the production pool is untouched."""
    source = create_signal("Source", np.arange(100.0), np.sin(np.arange(100.0)))
    param = GaussianParam.create(sigma=2.0)
    production_pool = base.POOL
    executor = PreviewExecutor()
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        failed = executor.submit(lambda: None, ()).result(timeout=60)
        assert failed.error_msg
        output = executor.submit(gaussian_filter, (source, param)).result(timeout=60)
        assert not output.error_msg, output.error_msg
        np.testing.assert_allclose(output.result.y, gaussian_filter(source, param).y)
        assert output.result.y.shape == source.y.shape
        future = executor.submit(slow_identity, (source, sender))
        assert receiver.poll(10)
        assert receiver.recv() is True
        start = time.monotonic()
        executor.close()
        assert time.monotonic() - start < 1.0
        executor.close(wait=True)
        assert time.monotonic() - start < 10.0
        assert future.result().cancelled
        assert executor._pool is None
        assert base.POOL is production_pool
    finally:
        executor.close(wait=True)
        receiver.close()
        sender.close()


def test_close_during_startup():
    """Closing immediately is idempotent, including a not-yet-created pool."""
    executor = PreviewExecutor()
    future = executor.submit(abs, (-1,))
    executor.close()
    executor.close(wait=True)
    assert future.done()
    assert executor._pool is None
