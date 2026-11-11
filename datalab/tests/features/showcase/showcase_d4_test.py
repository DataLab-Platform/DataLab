# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenario D4 tests

Scenario D4 (macros and remote control) is played in check mode (speed 0): a macro
runs in its own process, then a client process drives DataLab through the Web API.
The scenario must leave no trace behind: no macro, no entry in the macro recovery
cache, no server and no client process left running, no temporary file.
"""

from __future__ import annotations

import os
import os.path as osp
import socket
import tempfile
from pathlib import Path

import pytest

from datalab.gui.showcase import ShowcaseOptions, play_showcase
from datalab.gui.showcase.engine import CaptionBanner, ShowcaseContext
from datalab.gui.showcase.scenarios import d4_automation, d4_remote
from datalab.tests import datalab_test_app_context
from datalab.utils import macrorecovery


def _free_port() -> int:
    """Return a free TCP port of the local host"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(name="isolated")
def fixture_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the files written by the scenario in a temporary directory

    The Web API connection file, the macro recovery cache and the output file of
    the client are written to the temporary directory: they must not replace those
    of a DataLab instance in use. The Web API server listens on a free port.
    """
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("DATALAB_WEBAPI_PORT", str(_free_port()))
    recovery_path = str(tmp_path / macrorecovery.PENDING_FILENAME)
    monkeypatch.setattr(macrorecovery, "_path", lambda: recovery_path)
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    return tmp_path


def test_d4_leaves_no_trace(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D4 plays twice in a row, and leaves no trace behind"""
    saved: list[str] = []
    save_pending = macrorecovery.save_pending

    def spy_save_pending(uid: str, title: str, code: str) -> None:
        saved.append(title)
        save_pending(uid, title, code)

    monkeypatch.setattr(macrorecovery, "save_pending", spy_save_pending)
    connection_file = isolated / "datalab" / "webapi_connection.json"
    connection_file.parent.mkdir(parents=True, exist_ok=True)
    connection_file.write_text('{"url": "http://127.0.0.1:1", "token": "other"}')
    options = ShowcaseOptions(scenario_ids=["d4"], speed=0.0)
    with datalab_test_app_context(console=False) as win:
        rpc_running = win.remote_server.isRunning()
        for _index in range(2):
            results = play_showcase(win, options)
            assert [result.status for result in results] == ["done"], results
            # Macro created, then removed with its entry in the recovery cache
            assert d4_automation.MACRO_TITLE in saved
            assert len(win.macropanel) == 0
            assert not macrorecovery.load_pending()
            # Acquisitions received through the Web API, and their average
            titles = [obj.title for obj in win.signalpanel.objmodel]
            acquisitions = [t for t in titles if t.startswith("Acquisition ")]
            assert len(acquisitions) == d4_automation.ACQUISITION_COUNT
            assert len(titles) == d4_automation.MACRO_SIGNALS + len(acquisitions) + 1
            # Servers stopped (or left as found), client process stopped
            assert not win.get_webapi_status()["running"]
            assert win.remote_server.isRunning() == rpc_running
            leftovers = os.listdir(tempfile.gettempdir())
            assert not [n for n in leftovers if n.startswith(d4_remote.LOG_PREFIX)]
            # Connection file of another Web API server restored
            assert "other" in connection_file.read_text()
    assert osp.isfile(connection_file)


def test_d4_skipped_without_webapi(monkeypatch: pytest.MonkeyPatch) -> None:
    """D4 is skipped when the Web API is not available"""
    with datalab_test_app_context(console=False) as win:
        status = {"running": False, "url": None, "token": None, "available": False}
        monkeypatch.setattr(win, "get_webapi_status", lambda: status)
        results = play_showcase(win, ShowcaseOptions(scenario_ids=["d4"], speed=0.0))
        assert len(win.macropanel) == 0
    assert [result.status for result in results] == ["skipped"]
    assert "Web API" in results[0].message


def test_caption_above_raised_dock() -> None:
    """The caption gets back above a dock raised by DataLab (e.g. a new object)"""
    with datalab_test_app_context(console=False) as win:
        with ShowcaseContext(win, speed=0.01) as ctx:
            ctx.say("Caption", hold=0.0)
            caption = win.findChildren(CaptionBanner)[0]
            dock = win.docks[win.macropanel]
            dock.raise_()
            children = win.children()
            assert children.index(caption) < children.index(dock)
            # The caption is raised again by the guard timer of the context
            ctx.wait_until(lambda: False, 3 * ShowcaseContext.GUARD_INTERVAL_MS / 1e3)
            children = win.children()
            assert children.index(caption) > children.index(dock)
