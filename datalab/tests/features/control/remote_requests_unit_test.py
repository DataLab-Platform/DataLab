# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Remote requests synchronization tests

A remote command must be done when its XML-RPC call returns, or report an error,
including when it is sent as soon as DataLab accepts connections, i.e. while the
main window may still be set up (see DataLab Issue #398).
"""

# guitest: show

from __future__ import annotations

import tempfile
import threading
import xmlrpc.client
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sigima.objects import create_signal

from datalab.tests import close_datalab_background, datalab_in_background_context

if TYPE_CHECKING:
    from datalab.control.proxy import RemoteProxy


@pytest.mark.parametrize("mode", ["files", "directory"])
def test_remote_load_at_startup(tmp_path: Path, mode: str) -> None:
    """Files loaded remotely right after connecting are loaded when the call returns

    Args:
        tmp_path: Temporary directory
        mode: "files" to call ``load_from_files``, "directory" to call
         ``load_from_directory``
    """
    filename = tmp_path / "image.npy"
    np.save(filename, np.arange(6, dtype=np.uint16).reshape(2, 3))
    with datalab_in_background_context() as proxy:
        proxy.set_current_panel("image")
        if mode == "files":
            proxy.load_from_files([str(filename)])
        else:
            proxy.load_from_directory(str(tmp_path))
        assert len(proxy.get_object_uuids("image")) == 1


def test_remote_command_error() -> None:
    """Errors raised by remote commands are reported to the client"""
    with datalab_in_background_context() as proxy:
        with pytest.raises(xmlrpc.client.Fault, match="Unknown computation function"):
            proxy.calc("unknown_function")
        unknown = create_signal("Unknown signal", np.arange(3.0), np.zeros(3))
        with pytest.raises(xmlrpc.client.Fault, match="KeyError"):
            proxy.set_object(unknown)
        # The error must not be reported again by the next command
        proxy.set_current_panel("signal")
        assert proxy.get_current_panel() == "signal"


def test_remote_command_result() -> None:
    """The result of a remote command is not returned by the next command"""
    with datalab_in_background_context() as proxy:
        assert proxy.call_method("get_current_panel") == "signal"
        # No object: the result of the previous command must not be returned instead
        assert proxy.get_current_object_uuid() is None


def test_remote_close_during_call() -> None:
    """Closing DataLab while a remote call is pending does not hang DataLab"""
    errors: list[Exception] = []

    def close_datalab(proxy: RemoteProxy) -> None:
        """Close DataLab with a remote call waiting for DataLab to be ready"""
        try:
            proxy.call_method("close")
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(exc)

    with datalab_in_background_context() as proxy:
        thread = threading.Thread(target=close_datalab, args=(proxy,), daemon=True)
        thread.start()
        thread.join(20.0)
        hung = thread.is_alive()
        if hung:
            # Requesting a hung DataLab to close would hang the test as well
            close_datalab_background(request_close=False)
        assert not hung, "DataLab hung while closing during remote call"
        assert not errors


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        for load_mode in ("files", "directory"):
            test_remote_load_at_startup(Path(tmpdir), load_mode)
    test_remote_command_error()
    test_remote_command_result()
    test_remote_close_during_call()
