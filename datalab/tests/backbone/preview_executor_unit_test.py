"""Windows-compatible process lifecycle for speculative computations."""

from __future__ import annotations

import multiprocessing
import time

import numpy as np
from sigima.objects import create_signal
from sigima.params import GaussianParam
from sigima.proc.signal import gaussian_filter

from datalab.gui.processor import base
from datalab.gui.processor.preview import PreviewExecutor


def slow_identity(source, started):
    """Represent an expensive computation that must be cancellable."""
    started.send(True)
    time.sleep(30)
    return source


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
