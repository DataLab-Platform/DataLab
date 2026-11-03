# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Remote requests synchronization tests

A remote command must be done when its XML-RPC call returns, or report an error,
including when it is sent as soon as DataLab accepts connections, i.e. while the
main window may still be set up (see DataLab Issue #398).
"""

# guitest: show

from __future__ import annotations

import json
import tempfile
import threading
import urllib.error
import urllib.request
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


def test_remote_processing_error(tmp_path: Path) -> None:
    """Errors caught by error boundaries are reported to the client (Issue #404)

    Computation and file loading errors were shown in a message box, which crashed
    DataLab when it ran without display, instead of being reported to the client.

    Args:
        tmp_path: Temporary directory
    """
    filename = tmp_path / "invalid.npy"
    filename.write_text("Not a NumPy file", encoding="utf-8")
    with datalab_in_background_context() as proxy:
        proxy.add_signal("One point", np.array([0.0]), np.array([1.0]))
        proxy.select_objects([1], "signal")
        # The derivative of a one-point signal raises IndexError in Sigima
        with pytest.raises(xmlrpc.client.Fault, match="IndexError"):
            proxy.calc("derivative")
        with pytest.raises(xmlrpc.client.Fault):
            proxy.load_from_files([str(filename)])
        # DataLab is still running
        assert proxy.get_object_titles("signal") == ["One point"]


def test_remote_command_result() -> None:
    """The result of a remote command is not returned by the next command"""
    with datalab_in_background_context() as proxy:
        assert proxy.call_method("get_current_panel") == "signal"
        # No object: the result of the previous command must not be returned instead
        assert proxy.get_current_object_uuid() is None


def test_remote_add_group_returns_uuid() -> None:
    """The UUID returned by add_group lets the client fill the new group"""
    with datalab_in_background_context() as proxy:
        group_id = proxy.add_group("Scan 12.1", panel="signal")
        assert isinstance(group_id, str) and group_id
        signal = create_signal("I0", np.arange(3.0), np.ones(3))
        proxy.add_object(signal, group_id=group_id)
        assert len(proxy.get_object_uuids("signal", group=group_id)) == 1
        # The group UUID must not be returned by the next command
        proxy.set_current_panel("image")
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


def test_remote_webapi_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The Web API server started by a remote client serves requests (Issue #403)

    Its requests timed out after 30 seconds: the server was created in the XML-RPC
    server thread, which has no event loop, instead of the GUI thread.

    Args:
        tmp_path: Temporary directory
        monkeypatch: Fixture isolating the connection file of the Web API server
    """
    # The connection file written by the Web API server of the DataLab process must
    # not replace the one of a DataLab instance in use
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    with datalab_in_background_context() as proxy:
        info = proxy.start_webapi_server()
        try:
            assert proxy.get_webapi_status()["running"]
            request = urllib.request.Request(
                f"{info['url']}/api/v1/calc",
                data=json.dumps({"name": "unknown_function"}).encode(),
                headers={
                    "Authorization": f"Bearer {info['token']}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            # The request is served by the GUI thread, which reports at once that the
            # function is unknown (instead of HTTP 500 after 30 seconds)
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                with urllib.request.urlopen(request, timeout=20.0):
                    pass
            assert exc_info.value.code == 404
            assert "Unknown computation function" in exc_info.value.read().decode()
        finally:
            proxy.stop_webapi_server()
        assert not proxy.get_webapi_status()["running"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        for load_mode in ("files", "directory"):
            test_remote_load_at_startup(Path(tmpdir), load_mode)
    test_remote_command_error()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_remote_processing_error(Path(tmpdir))
    test_remote_command_result()
    test_remote_add_group_returns_uuid()
    test_remote_close_during_call()
    with tempfile.TemporaryDirectory() as tmpdir, pytest.MonkeyPatch.context() as mp:
        test_remote_webapi_server(Path(tmpdir), mp)
