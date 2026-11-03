# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Remote request error tests (in-process)

Errors of remote requests (XML-RPC, Web API) must be reported to the client and shown
in the status bar, never in a message box: no user waits for it, and showing a
message box crashes DataLab when it runs without display (see DataLab Issue #404).

These tests behave as outside pytest (production error boundaries), so that they do
not depend on whether the DataLab process imports pytest: message boxes are recorded
instead of being shown.
"""

# guitest: show

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import h5py
import numpy as np
import pytest
from qtpy import QtWidgets as QW
from sigima.objects import create_signal
from sigimax.utils import qthelpers as sgmx_qth

from datalab.env import execenv
from datalab.gui.h5io import H5InputOutput
from datalab.gui.main import DLMainWindow
from datalab.gui.processor.base import FeatureNotFoundError
from datalab.tests import datalab_test_app_context
from datalab.utils import qthelpers as dl_qth
from datalab.webapi.adapter import WorkspaceAdapter

FAILED_REQUEST = "Remote request failed"


@pytest.fixture(name="message_boxes")
def fixture_message_boxes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record message boxes instead of showing them

    Returns:
        Texts of the message boxes shown during the test
    """
    texts: list[str] = []

    def record(_parent: QW.QWidget, _title: str, text: str, *_args) -> int:
        texts.append(text)
        return QW.QMessageBox.Ok

    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QW.QMessageBox, name, staticmethod(record))
    return texts


@pytest.fixture(name="win")
def fixture_win(monkeypatch: pytest.MonkeyPatch) -> Generator[DLMainWindow, None, None]:
    """DataLab main window with production error boundaries, holding a signal on
    which the derivative fails (IndexError raised by Sigima)

    Returns:
        DataLab main window
    """
    with datalab_test_app_context(exec_loop=False) as win:
        win.signalpanel.add_object(
            create_signal("One point", np.array([0.0]), np.array([1.0]))
        )
        win.statusBar().clearMessage()
        # Error boundaries behave as outside tests (no re-raise because of pytest)
        for module in (sgmx_qth, dl_qth):
            monkeypatch.setattr(module, "is_running_tests", lambda: False)
        yield win


def run_xmlrpc_command(
    win: DLMainWindow, command: Callable[..., object], *args
) -> Exception | None:
    """Run a command as requested by an XML-RPC client

    Args:
        win: DataLab main window
        command: Main window method running the command
        args: Command arguments

    Returns:
        Exception to be reported to the client, if any
    """
    win.remote_server.exception = None
    win.run_remote_command(command, *args)
    exc, win.remote_server.exception = win.remote_server.exception, None
    return exc


def assert_reported_in_status_bar(win: DLMainWindow, text: str) -> None:
    """Check that a failed remote request is shown in the status bar

    Args:
        win: DataLab main window
        text: Text expected in the status bar message
    """
    message = win.statusBar().currentMessage()
    assert FAILED_REQUEST in message and text in message, message


def test_gui_computation_error_shown(win: DLMainWindow, message_boxes: list[str]):
    """A computation error is still shown in a message box outside remote requests"""
    win.calc("derivative")
    assert len(message_boxes) == 1
    assert FAILED_REQUEST not in win.statusBar().currentMessage()


def test_xmlrpc_computation_error(win: DLMainWindow, message_boxes: list[str]):
    """A computation error is reported to the XML-RPC client"""
    exc = run_xmlrpc_command(win, win.calc, "derivative")
    assert isinstance(exc, IndexError)
    assert not message_boxes
    assert_reported_in_status_bar(win, str(exc))
    # The error does not change the behavior of the next interactive action
    win.calc("derivative")
    assert len(message_boxes) == 1


def test_xmlrpc_call_method_error(win: DLMainWindow, message_boxes: list[str]):
    """A computation error is reported to the XML-RPC client (``call_method``)"""
    win.remote_server.exception = None
    win.call_method_slot("calc", ["derivative"], None, {})
    assert isinstance(win.remote_server.exception, IndexError)
    win.remote_server.exception = None
    assert not message_boxes
    assert_reported_in_status_bar(win, "out of bounds")


def test_xmlrpc_unknown_computation(win: DLMainWindow, message_boxes: list[str]):
    """An unknown computation is reported as such to the XML-RPC client"""
    exc = run_xmlrpc_command(win, win.calc, "unknown_function")
    assert isinstance(exc, FeatureNotFoundError) and isinstance(exc, ValueError)
    assert "Unknown computation function" in str(exc)
    assert not message_boxes


def test_webapi_computation_error(win: DLMainWindow, message_boxes: list[str]):
    """A computation error is reported to the Web API client"""
    adapter = WorkspaceAdapter(win)
    with pytest.raises(RuntimeError, match="Computation 'derivative' failed"):
        adapter.calc("derivative")
    assert not message_boxes
    assert_reported_in_status_bar(win, "Computation 'derivative' failed")
    # Unknown functions are told apart from computation errors (HTTP 404)
    with pytest.raises(FeatureNotFoundError):
        adapter.calc("unknown_function")


def test_xmlrpc_load_error(win: DLMainWindow, message_boxes: list[str], tmp_path: Path):
    """A file that cannot be read is reported to the XML-RPC client"""
    filename = tmp_path / "invalid.npy"
    filename.write_text("Not a NumPy file", encoding="utf-8")
    exc = run_xmlrpc_command(win, win.load_from_files, [str(filename)])
    assert exc is not None
    assert not message_boxes
    assert_reported_in_status_bar(win, str(exc).strip().splitlines()[0])
    # Interactive loading still shows the error in a message box
    win.load_from_files([str(filename)])
    assert len(message_boxes) == 1


def test_xmlrpc_hdf5_import_errors(
    win: DLMainWindow,
    message_boxes: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Datasets that cannot be imported are reported to the XML-RPC client"""
    filename = tmp_path / "data.h5"
    with h5py.File(filename, "w") as h5file:
        h5file.create_dataset("image", data=np.arange(12.0).reshape(3, 4))

    def fail_import(*_args) -> None:
        raise ValueError("Unsupported dataset")

    monkeypatch.setattr(
        H5InputOutput, "_H5InputOutput__add_object_from_node", fail_import
    )
    # Outside unattended mode, import errors are shown unless reported by the caller
    with execenv.context(unattended=False):
        exc = run_xmlrpc_command(win, win.open_h5_files, [str(filename)], True, True)
        assert isinstance(exc, ValueError) and "Unsupported dataset" in str(exc)
        assert not message_boxes
        assert_reported_in_status_bar(win, "Unsupported dataset")
        # Interactive import still shows the errors in a message box
        win.open_h5_files([str(filename)], True, True)
    assert len(message_boxes) == 1 and "Unsupported dataset" in message_boxes[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
