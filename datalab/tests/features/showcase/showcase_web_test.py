# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase DataLab Web interlude tests

URL of DataLab Web, discovery of the browser, playlist, and interlude played in
check mode with a fake browser (a Python script exiting after a while).
"""

from __future__ import annotations

import json
import os.path as osp
import sys
import time
from collections.abc import Callable

import pytest

from datalab.config import Conf
from datalab.gui.showcase import ShowcaseDialog, ShowcaseOptions, ShowcasePlayer
from datalab.gui.showcase.engine import ScenarioSkipped, ShowcaseScenario
from datalab.gui.showcase.player import build_playlist
from datalab.gui.showcase.scenarios import get_native_scenarios, web_interlude
from datalab.tests import datalab_test_app_context

#: Fake browser: records its arguments, then exits after a while (with a duration
#: "relaunch:<duration>", starts again in a new process, then exits at once)
FAKE_BROWSER = """
import json, subprocess, sys, time
record, duration, *arguments = sys.argv[1:]
if duration.startswith("relaunch:"):
    command = [sys.executable, sys.argv[0], record, duration[9:], *arguments]
    subprocess.Popen(command, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    sys.exit(0)
with open(record, "w", encoding="utf-8") as fobj:
    json.dump(arguments, fobj)
time.sleep(float(duration))
"""


def test_web_url() -> None:
    """The URL plays one pass of the DataLab Web showcase, then closes the window"""
    build = web_interlude.build_web_url
    assert (
        build("https://datalab-platform.com/web/", None, "fr", 1.5)
        == "https://datalab-platform.com/web/?showcase=1&close=1&lang=fr&speed=1.5"
    )
    # Parameters of the URL are kept, except those set by the interlude
    assert (
        build("http://localhost:5173/?showcase=w1&loop=1&runtime=main", None, "en", 0)
        == "http://localhost:5173/?runtime=main&showcase=w1&close=1&lang=en"
    )
    assert build("http://localhost:5173?showcase=w1", ["w2", "w3"], "en", 1.0) == (
        "http://localhost:5173?showcase=w2,w3&close=1&lang=en&speed=1"
    )
    for url in ("localhost:5173", "datalab-platform.com/web", "ftp://host/web"):
        with pytest.raises(ScenarioSkipped):
            build(url)
    assert web_interlude.parse_ids("") is None
    assert web_interlude.parse_ids("all") is None
    assert web_interlude.parse_ids(" w1, ,w3 ") == ["w1", "w3"]
    assert web_interlude.get_ui_language() in web_interlude.WEB_LANGUAGES


def test_browser_command() -> None:
    """The browser shows DataLab Web in an app window, with a dedicated profile"""
    command = web_interlude.browser_command(
        "msedge.exe", "https://host/?showcase=1", "profile", True, (1920, 0, 800, 600)
    )
    assert command[:3] == [
        "msedge.exe",
        "--app=https://host/?showcase=1",
        "--user-data-dir=profile",
    ]
    for option in (
        "--no-first-run",
        "--disable-background-timer-throttling",
        "--window-position=1920,0",
        "--window-size=800,600",
        "--start-fullscreen",
    ):
        assert option in command
    command = web_interlude.browser_command("chrome", "https://host/", "profile")
    assert "--start-maximized" in command
    assert not any(arg.startswith("--window-position") for arg in command)


def test_browser_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Google Chrome is preferred to Microsoft Edge, then Chromium-based browsers
    are looked for on the PATH"""
    module = web_interlude
    edge = tmp_path / "Microsoft" / "Edge" / "Application" / "msedge.exe"
    edge.parent.mkdir(parents=True)
    edge.write_text("")
    registered: dict[str, str] = {}
    on_path: dict[str, str] = {}
    monkeypatch.setattr(module, "registry_app_path", registered.get)
    monkeypatch.setattr(module.shutil, "which", on_path.get)
    for variable in module.WINDOWS_PROGRAM_FOLDERS:
        monkeypatch.delenv(variable, raising=False)
    assert module.find_browser("win32") is None
    monkeypatch.setenv("PROGRAMFILES(X86)", str(tmp_path))
    assert module.find_browser("win32") == str(edge)
    chrome = tmp_path / "chrome.exe"
    chrome.write_text("")
    registered["chrome.exe"] = str(chrome)
    assert module.find_browser("win32") == str(chrome)
    registered["chrome.exe"] = str(tmp_path / "uninstalled" / "chrome.exe")
    assert module.find_browser("win32") == str(edge)
    assert module.find_browser("linux") is None
    on_path["chromium"] = "/usr/bin/chromium"
    assert module.find_browser("linux") == "/usr/bin/chromium"
    on_path["google-chrome"] = "/usr/bin/google-chrome"
    assert module.find_browser("linux") == "/usr/bin/google-chrome"


def test_playlist() -> None:
    """The interlude ends the playlist when a DataLab Web URL is given, or is played
    where its ID is placed"""
    scenarios = get_native_scenarios()
    ids = [scenario.id for scenario in scenarios]

    def playlist(**kwargs) -> list[str]:
        options = ShowcaseOptions(**kwargs)
        return [scenario.id for scenario in build_playlist(scenarios, options)]

    assert playlist() == ids
    assert playlist(web_url="http://host/") == ids + ["web"]
    assert playlist(scenario_ids=["d3", "d1"], web_url="http://host/") == [
        "d3",
        "d1",
        "web",
    ]
    assert playlist(scenario_ids=["web", "d2"], web_url="http://host/") == [
        "web",
        "d2",
    ]
    # Placed explicitly without URL: it is skipped when played (see below)
    assert playlist(scenario_ids=["d2", "web"]) == ["d2", "web"]


def _play(win, scenario: ShowcaseScenario) -> tuple[str, str]:
    """Play a scenario in check mode, and return its status and message"""
    options = ShowcaseOptions(speed=0.0)
    results = ShowcasePlayer(win, [scenario], options).play()
    return results[0].status, results[0].message


def test_interlude_is_skipped_without_url_or_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without DataLab Web URL, or without browser, the interlude is skipped"""
    monkeypatch.setattr(web_interlude, "find_browser", lambda: None)
    with datalab_test_app_context(console=False) as win:
        status, message = _play(win, web_interlude.create_scenario(None))
        assert status == "skipped" and "URL" in message
        status, message = _play(win, web_interlude.create_scenario("http://host/"))
        assert status == "skipped" and "browser" in message
        status, message = _play(win, web_interlude.create_scenario("localhost"))
        assert status == "skipped" and "Invalid" in message
        assert not win.isMinimized()


@pytest.fixture(name="fake_browser")
def fixture_fake_browser(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> Callable[[float], str]:
    """Replace the browser by a Python script exiting after a given duration

    Returns:
        Function setting the duration of the fake browser (s), and returning the
         file where the fake browser records its arguments
    """
    script = tmp_path / "fake_browser.py"
    script.write_text(FAKE_BROWSER, encoding="utf-8")
    record = tmp_path / "arguments.json"
    profile = tmp_path / "profile"
    monkeypatch.setattr(web_interlude, "CLOSE_TIMEOUT", 1.0)
    monkeypatch.setattr(web_interlude, "get_profile_dir", lambda: str(profile))
    monkeypatch.setattr(web_interlude, "find_browser", lambda: "fake-browser")
    original_command = web_interlude.browser_command

    def set_duration(duration: float | str) -> str:
        def browser_command(browser: str, *args, **kwargs) -> list[str]:
            command = original_command(browser, *args, **kwargs)
            fake = [sys.executable, str(script), str(record), str(duration)]
            return fake + command[1:]

        monkeypatch.setattr(web_interlude, "browser_command", browser_command)
        return str(record)

    return set_duration


def _record_window_states(
    monkeypatch: pytest.MonkeyPatch, action: Callable | None = None
) -> list[bool]:
    """Record whether the main window is minimized while waiting for the browser

    Args:
        monkeypatch: pytest monkeypatch fixture
        action: Called with the showcase context before waiting

    Returns:
        List of the recorded states (True: minimized)
    """
    states: list[bool] = []
    original_wait = web_interlude.wait_for_browser

    def wait_for_browser(ctx, browser, timeout) -> bool:
        states.append(ctx.win.isMinimized())
        if action is not None:
            action(ctx)
        return original_wait(ctx, browser, timeout)

    monkeypatch.setattr(web_interlude, "wait_for_browser", wait_for_browser)
    return states


def test_interlude_waits_for_the_browser(
    monkeypatch: pytest.MonkeyPatch, fake_browser: Callable[[float | str], str]
) -> None:
    """The main window is minimized while the browser shows DataLab Web, then shown
    again once the browser exits"""
    record = fake_browser(1.0)
    states = _record_window_states(monkeypatch)
    with datalab_test_app_context(console=False) as win:
        scenario = web_interlude.create_scenario("https://host/web/", ["w1"])
        assert _play(win, scenario) == ("done", "")
        assert states == [True]
        assert not win.isMinimized()
    with open(record, encoding="utf-8") as fobj:
        arguments = json.load(fobj)
    lang = web_interlude.get_ui_language()
    assert arguments[0] == f"--app=https://host/web/?showcase=w1&close=1&lang={lang}"
    profile = web_interlude.get_profile_dir()
    assert f"--user-data-dir={profile}" in arguments
    # The profile does not offer to translate DataLab Web
    with open(osp.join(profile, "Default", "Preferences"), encoding="utf-8") as fobj:
        assert json.load(fobj)["translate"]["enabled"] is False
    assert not web_interlude.BrowserProcess.running


def test_interlude_waits_for_a_relaunched_browser(
    fake_browser: Callable[[float | str], str],
) -> None:
    """A browser starting again in a new process with the dedicated profile (e.g.
    Microsoft Edge on Windows) is waited for"""
    record = fake_browser("relaunch:2.0")
    with datalab_test_app_context(console=False) as win:
        scenario = web_interlude.create_scenario("https://host/web/")
        start = time.monotonic()
        assert _play(win, scenario) == ("done", "")
        assert time.monotonic() - start >= 2.0
    assert osp.isfile(record)
    assert not web_interlude.find_profile_processes(web_interlude.get_profile_dir())


def test_profile_processes(tmp_path) -> None:
    """The processes of the dedicated profile are recognized from their command
    line, and the preferences of the profile are merged"""
    profile = str(tmp_path / "profile")
    uses = web_interlude.uses_profile
    assert uses(["msedge.exe", f"--user-data-dir={profile}"], profile)
    assert uses(["msedge.exe", f'--user-data-dir="{profile}{osp.sep}"'], profile)
    assert not uses(["msedge.exe", f"--user-data-dir={profile}-other"], profile)
    assert not uses(["msedge.exe", "--app=https://host/"], profile)
    preferences = tmp_path / "profile" / "Default" / "Preferences"
    preferences.parent.mkdir(parents=True)
    preferences.write_text(json.dumps({"translate": {"x": 1}, "other": 2}))
    web_interlude.prepare_profile(profile)
    assert json.loads(preferences.read_text()) == {
        "translate": {"x": 1, "enabled": False},
        "other": 2,
    }


@pytest.mark.parametrize("ending", ["timeout", "stop"])
def test_interlude_closes_the_browser(
    ending: str,
    monkeypatch: pytest.MonkeyPatch,
    fake_browser: Callable[[float | str], str],
) -> None:
    """The browser is closed after the timeout, or when the operator stops the
    showcase: the main window is shown again"""
    fake_browser(60.0)
    started: list[web_interlude.BrowserProcess] = []
    original_init = web_interlude.BrowserProcess.__init__

    def init(self, command: list[str], profile_dir: str) -> None:
        original_init(self, command, profile_dir)
        started.append(self)

    monkeypatch.setattr(web_interlude.BrowserProcess, "__init__", init)
    action = (lambda ctx: ctx.request_stop()) if ending == "stop" else None
    states = _record_window_states(monkeypatch, action)
    with datalab_test_app_context(console=False) as win:
        scenario = web_interlude.create_scenario("https://host/", timeout=1.0)
        status, message = _play(win, scenario)
        if ending == "timeout":
            assert status == "failed" and "did not close" in message
        else:
            assert status == "done"
        assert states == [True] and not win.isMinimized()
    (browser,) = started
    assert not browser.is_running()
    assert not web_interlude.find_profile_processes(browser.profile_dir)
    assert not web_interlude.BrowserProcess.running


def test_dialog_includes_datalab_web() -> None:
    """The Showcase mode dialog includes DataLab Web on request, and remembers it"""
    with (
        Conf.showcase_web_enabled.context(False),
        Conf.showcase_web_url.context(""),
        datalab_test_app_context(console=False) as win,
    ):
        dialog = ShowcaseDialog(win, get_native_scenarios())
        assert not dialog.web_check.isChecked()
        assert not dialog.web_url_edit.isEnabled()
        dialog.accept()
        assert dialog.options.web_url is None
        dialog = ShowcaseDialog(win, get_native_scenarios())
        dialog.web_check.setChecked(True)
        assert dialog.web_url_edit.isEnabled()
        dialog.accept()
        assert dialog.options.web_url == web_interlude.DEFAULT_URL
        assert Conf.showcase_web_enabled.get() and Conf.showcase_web_url.get() == ""
        dialog = ShowcaseDialog(win, get_native_scenarios())
        assert dialog.web_check.isChecked()
        dialog.web_url_edit.setText(" http://localhost:5173/ ")
        dialog.accept()
        assert dialog.options.web_url == "http://localhost:5173/"
        assert Conf.showcase_web_url.get() == "http://localhost:5173/"
