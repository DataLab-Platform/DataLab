# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""Regression tests for Web API availability and dependency diagnostics."""

from __future__ import annotations

import builtins
import importlib.util
import os
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW

import datalab.gui.main as main
import datalab.webapi as webapi
import datalab.webapi.actions as actions
from datalab.webapi.controller import WebApiController

DIAGNOSTIC = (
    "Web API is unavailable. FastAPI, Uvicorn and Pydantic are required "
    "dependencies. Check or repair your DataLab installation using the original "
    "installation method or package manager. For pip installations, run "
    "python -m pip check in DataLab's Python environment to check installed "
    "dependency requirements."
)


@pytest.fixture
def qt_app() -> Generator[QW.QApplication, None, None]:
    """Provide an application without starting a DataLab window or server."""
    with qt_app_context(exec_loop=False) as app:
        yield app


def _window_stub() -> SimpleNamespace:
    """Provide the state needed by the remote-controlled method decorator."""
    return SimpleNamespace(ready_flag=True, SIG_READY=Mock())


def test_diagnostic_gives_non_mutating_guidance() -> None:
    """Name requirements and suggest checking them without modifying packages."""
    message = webapi.webapi_unavailable_message()
    assert message == DIAGNOSTIC
    assert "FastAPI, Uvicorn and Pydantic are required dependencies" in message
    assert "original installation method or package manager" in message
    assert "python -m pip check" in message
    assert "python -m pip install" not in message


@pytest.mark.parametrize(
    ("blocked_import", "failed_module"),
    [
        ("fastapi", "fastapi"),
        ("pydantic", "pydantic"),
        ("uvicorn", "uvicorn"),
        ("fastapi", "starlette.concurrency"),
    ],
)
def test_dependency_import_failures(
    monkeypatch: pytest.MonkeyPatch, blocked_import: str, failed_module: str
) -> None:
    """Report direct and transitive failures without claiming a package is missing."""
    if blocked_import == failed_module:
        failure = ModuleNotFoundError(
            f"No module named '{failed_module}'", name=failed_module
        )
    else:
        failure = ImportError(
            f"cannot import name 'broken' from '{failed_module}'", name=failed_module
        )
    spec = importlib.util.spec_from_file_location("_webapi_probe", webapi.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == blocked_import:
            raise failure
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", failing_import)
        spec.loader.exec_module(probe)

    assert not probe.WEBAPI_AVAILABLE
    with pytest.raises(ImportError) as exc_info:
        probe.get_webapi_controller()
    assert str(exc_info.value) == DIAGNOSTIC
    assert exc_info.value.__cause__ is failure


def test_main_window_starts_when_imports_fail(tmp_path: Path) -> None:
    """A fresh application stays usable when a Web API import is incompatible."""
    code = """
import builtins
import os

original_import = builtins.__import__

def failing_import(name, *args, **kwargs):
    if name == "fastapi":
        raise ImportError("cannot import name 'broken' from 'starlette.concurrency'")
    return original_import(name, *args, **kwargs)

builtins.__import__ = failing_import
from datalab.config import Conf, ensure_initialized
from datalab.env import execenv
from datalab.gui.main import DLMainWindow
from datalab.utils.qthelpers import datalab_app_context

execenv.unattended = True
execenv.verbose = "quiet"
ensure_initialized(load_user_config=False)
Conf.rpc_server_enabled.set(False)
Conf.faulthandler_enabled.set(False)
Conf.plugins_enabled.set(False)
with datalab_app_context(exec_loop=False, enable_logs=False):
    window = DLMainWindow(console=False)
    try:
        assert not window.webapi_actions._start_action.isEnabled()
        assert window.webapi_actions._status_action.toolTip() == os.environ["EXPECTED"]
        assert window.get_webapi_status() == {
            "running": False, "url": None, "token": None, "available": False
        }
        try:
            window.start_webapi_server()
        except RuntimeError as error:
            assert str(error) == os.environ["EXPECTED"]
        else:
            raise AssertionError("Starting the unavailable Web API succeeded")
    finally:
        window.set_modified(False)
        window.close()
print("Application remained usable")
"""
    env = os.environ.copy()
    env.update(
        APPDATA=str(tmp_path),
        LOCALAPPDATA=str(tmp_path),
        USERPROFILE=str(tmp_path),
        HOME=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path),
        TEMP=str(tmp_path),
        TMP=str(tmp_path),
        QT_QPA_PLATFORM="offscreen",
        LANG="en_US.UTF-8",
        PYTHONIOENCODING="utf-8",
        EXPECTED=DIAGNOSTIC,
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Application remained usable" in result.stdout


def test_controller_import_failure_keeps_its_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later controller import failure gets the same advice and retains its cause."""
    monkeypatch.setattr(webapi, "WEBAPI_AVAILABLE", True)
    monkeypatch.setattr(webapi, "_CONTROLLER_INSTANCE", None)
    failure = ImportError("cannot import name 'broken' from 'qtpy'")
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "datalab.webapi.controller":
            raise failure
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", failing_import)
        with pytest.raises(ImportError) as exc_info:
            webapi.get_webapi_controller()
    assert str(exc_info.value) == DIAGNOSTIC
    assert exc_info.value.__cause__ is failure


@pytest.mark.usefixtures("qt_app")
@pytest.mark.parametrize("failure_path", ["dependencies", "controller"])
def test_unavailable_action_shows_translatable_guidance(
    monkeypatch: pytest.MonkeyPatch, failure_path: str
) -> None:
    """The disabled menu advertises and exposes the translated repair advice."""
    monkeypatch.setattr(actions, "_", lambda message: f"translated: {message}")
    monkeypatch.setattr(webapi, "WEBAPI_AVAILABLE", failure_path == "controller")
    if failure_path == "controller":
        monkeypatch.setattr(
            webapi, "get_webapi_controller", Mock(side_effect=ImportError("broken"))
        )
    window = QW.QMainWindow()
    web_actions = actions.WebApiActions(window)
    menu = web_actions.create_menu(QW.QMenu(window))

    assert web_actions._controller is None
    assert not web_actions._start_action.isEnabled()
    assert not web_actions._status_action.isEnabled()
    assert web_actions._status_action.text() == (
        "translated: Web API unavailable (hover for details)"
    )
    assert web_actions._status_action.toolTip() == f"translated: {DIAGNOSTIC}"
    assert menu.toolTipsVisible()
    web_actions._start_server()
    warning = Mock()
    question = Mock(return_value=QW.QMessageBox.No)
    monkeypatch.setattr(QW.QMessageBox, "warning", warning)
    monkeypatch.setattr(QW.QMessageBox, "question", question)
    web_actions.start_server_from_status_widget()
    warning.assert_called_once_with(
        window, actions.APP_NAME, f"translated: {DIAGNOSTIC}"
    )
    question.assert_not_called()
    web_actions.cleanup()
    window.close()


@pytest.mark.usefixtures("qt_app")
def test_start_action_reports_late_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An import failing during server startup receives the same translated advice."""
    failure = ImportError("cannot import name 'broken' from 'starlette'")
    controller = Mock(is_running=False)
    controller.start.side_effect = failure
    monkeypatch.setattr(webapi, "WEBAPI_AVAILABLE", True)
    monkeypatch.setattr(webapi, "get_webapi_controller", Mock(return_value=controller))
    monkeypatch.setattr(actions, "_", lambda message: f"translated: {message}")
    critical = Mock()
    monkeypatch.setattr(QW.QMessageBox, "critical", critical)
    window = QW.QMainWindow()
    web_actions = actions.WebApiActions(window)

    web_actions._start_server()

    message = critical.call_args.args[2]
    assert f"translated: {DIAGNOSTIC}" in message
    assert str(failure) in message
    web_actions.cleanup()
    window.close()


@pytest.mark.usefixtures("qt_app")
def test_control_api_when_dependencies_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting fails clearly while status and stop remain safe."""
    monkeypatch.setattr(main, "WEBAPI_AVAILABLE", False)
    get_controller = Mock()
    monkeypatch.setattr(main, "get_webapi_controller", get_controller)
    window = _window_stub()

    with pytest.raises(RuntimeError) as exc_info:
        main.DLMainWindow.start_webapi_server(window)
    assert str(exc_info.value) == DIAGNOSTIC
    assert main.DLMainWindow.get_webapi_status(window) == {
        "running": False,
        "url": None,
        "token": None,
        "available": False,
    }
    assert main.DLMainWindow.stop_webapi_server(window) is None
    get_controller.assert_not_called()
    assert window.ready_flag
    assert window.SIG_READY.emit.call_count == 3


@pytest.mark.usefixtures("qt_app")
@pytest.mark.parametrize("failure_path", ["controller", "start"])
def test_control_api_reports_late_import_failures(
    monkeypatch: pytest.MonkeyPatch, failure_path: str
) -> None:
    """A later import error is actionable and retains its original exception."""
    failure = ImportError("cannot import name 'broken' from 'starlette'")
    controller = Mock()
    if failure_path == "controller":
        get_controller = Mock(side_effect=failure)
    else:
        controller.start.side_effect = failure
        get_controller = Mock(return_value=controller)
    monkeypatch.setattr(main, "WEBAPI_AVAILABLE", True)
    monkeypatch.setattr(main, "get_webapi_controller", get_controller)

    with pytest.raises(RuntimeError) as exc_info:
        main.DLMainWindow.start_webapi_server(_window_stub())
    assert str(exc_info.value) == DIAGNOSTIC
    assert exc_info.value.__cause__ is failure


@pytest.mark.usefixtures("qt_app")
def test_control_api_handles_controller_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status and stop remain safe if the controller itself cannot be imported."""
    monkeypatch.setattr(main, "WEBAPI_AVAILABLE", True)
    monkeypatch.setattr(
        main, "get_webapi_controller", Mock(side_effect=ImportError("broken"))
    )
    window = _window_stub()

    assert main.DLMainWindow.get_webapi_status(window) == {
        "running": False,
        "url": None,
        "token": None,
        "available": False,
    }
    assert main.DLMainWindow.stop_webapi_server(window) is None


@pytest.mark.usefixtures("qt_app")
def test_available_controller_is_a_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Required dependencies enable the real controller without starting a server."""
    assert webapi.WEBAPI_AVAILABLE
    monkeypatch.setattr(webapi, "_CONTROLLER_INSTANCE", None)
    controller = webapi.get_webapi_controller()
    assert isinstance(controller, WebApiController)
    assert webapi.get_webapi_controller() is controller


@pytest.mark.usefixtures("qt_app")
def test_available_action_and_control_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """The menu and control API still use the available controller normally."""
    controller = Mock(is_running=False)
    controller.start.return_value = ("http://127.0.0.1:18080", "token")
    controller.get_connection_info.return_value = {
        "running": True,
        "url": "http://127.0.0.1:18080",
        "token": "token",
    }
    monkeypatch.setattr(webapi, "WEBAPI_AVAILABLE", True)
    monkeypatch.setattr(webapi, "get_webapi_controller", Mock(return_value=controller))
    monkeypatch.setattr(main, "WEBAPI_AVAILABLE", True)
    monkeypatch.setattr(main, "get_webapi_controller", Mock(return_value=controller))

    gui_window = QW.QMainWindow()
    web_actions = actions.WebApiActions(gui_window)
    show_dialog = Mock()
    monkeypatch.setattr(web_actions, "_show_connection_dialog", show_dialog)
    assert web_actions._start_action.isEnabled()
    assert web_actions._status_action.text() == actions._("Status: Not running")
    web_actions._start_server()
    show_dialog.assert_called_once_with("http://127.0.0.1:18080", "token")
    monkeypatch.setattr(
        QW.QMessageBox, "question", Mock(return_value=QW.QMessageBox.Yes)
    )
    start_from_status = Mock()
    monkeypatch.setattr(web_actions, "_start_server", start_from_status)
    web_actions.start_server_from_status_widget()
    start_from_status.assert_called_once_with()
    web_actions.cleanup()
    gui_window.close()

    control_window = _window_stub()
    assert main.DLMainWindow.start_webapi_server(
        control_window, host="127.0.0.1", port=18080
    ) == {"url": "http://127.0.0.1:18080", "token": "token"}
    controller.start.assert_called_with(host="127.0.0.1", port=18080)
    assert main.DLMainWindow.get_webapi_status(control_window) == {
        "running": True,
        "url": "http://127.0.0.1:18080",
        "token": "token",
        "available": True,
    }
    main.DLMainWindow.stop_webapi_server(control_window)
    controller.stop.assert_called_once_with()
