# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests of the shared execution layer of the tools: parameter values,
main thread submission and computation messages.
"""

# pylint: disable=protected-access

from __future__ import annotations

import threading

import pytest
import sigima.params as sp
from qtpy import QtWidgets as QW

from datalab.aiassistant.tools import builtin
from datalab.gui.processor import base
from datalab.gui.processor.catcher import CompOut
from datalab.tests import datalab_test_app_context
from datalab.webapi.adapter import MainThreadExecutor


def test_set_param_values_checks_every_value() -> None:
    """Values are converted when lossless and rejected with the valid ones."""
    param = sp.FWHMParam()
    builtin._set_param_values(param, {"method": "gauss", "xmin": 1})
    assert param.method == "gauss"
    assert isinstance(param.xmin, float) and param.xmin == 1.0
    with pytest.raises(ValueError, match="valid parameters: method, xmax, xmin"):
        builtin._set_param_values(param, {"sigma": 2})
    with pytest.raises(ValueError, match="Invalid value 'spline'"):
        builtin._set_param_values(param, {"method": "spline"})
    moving = sp.MovingAverageParam()
    builtin._set_param_values(moving, {"n": 5.0})
    assert moving.n == 5 and isinstance(moving.n, int)
    with pytest.raises(ValueError, match="parameter 'n'"):
        builtin._set_param_values(moving, {"n": 2.5})


def test_submit_cancelled_before_start_never_runs() -> None:
    """A future cancelled before the main thread runs it is skipped."""
    with datalab_test_app_context(console=False):
        executor = MainThreadExecutor()
        calls = []
        futures = {}

        def submit_from_worker() -> None:
            futures["cancelled"] = executor.submit(lambda: calls.append("cancelled"))
            futures["cancel_ok"] = futures["cancelled"].cancel()
            futures["done"] = executor.submit(lambda: calls.append("done") or 42)

        worker = threading.Thread(target=submit_from_worker)
        worker.start()
        worker.join()
        assert futures["cancel_ok"]
        done = futures["done"]
        while not done.done():
            QW.QApplication.processEvents()
        assert done.result() == 42
        assert calls == ["done"]


def test_computation_messages_are_collected_then_released() -> None:
    """Messages go to the active collector instead of message boxes."""
    with datalab_test_app_context(console=False) as win:
        processor = win.signalpanel.processor
        failed = CompOut(error_msg="boom")
        warned = CompOut(result="value", warning_msg="careful")
        with base.computation_messages_collected() as messages:
            assert processor.handle_output(failed, "Computing: A", None) is None
            assert processor.handle_output(warned, "Computing: B", None) == "value"
        assert messages.errors == ["Computing: A: boom"]
        assert messages.warnings == ["Computing: B: careful"]
    assert base._COMPUTATION_MESSAGES.get() is None
