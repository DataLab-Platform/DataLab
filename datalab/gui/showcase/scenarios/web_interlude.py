# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase interlude: DataLab Web
===============================

For booths with a single screen, the showcase hands over to DataLab Web for one
pass of its own showcase, in a browser window, then resumes once the window is
closed.

DataLab Web plays its showcase when opened with the ``showcase`` URL parameter, and
closes its window after one pass with ``close=1``. The browser is a Chromium-based
one (Google Chrome, Microsoft Edge or Chromium), started in *app* mode (a window
without tabs nor address bar) with a dedicated profile, in the DataLab
configuration folder: the browser process ends with its window, and the cache of
DataLab Web (Python runtime and packages) is kept from one interlude to the next.

The interlude is not one of the demos listed in the **Showcase mode** dialog: it
ends each loop when a DataLab Web URL is given (**Include DataLab Web** option of
the dialog, ``--showcase-web`` command-line option), or it is played where its ID
(``web``) is placed in the list of the scenarios to play.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from functools import partial
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psutil
from qtpy import QtCore as QC

from datalab.config import Conf, _, get_config_path
from datalab.env import SHOWCASE_WEB_URL
from datalab.gui.showcase.engine import (
    ScenarioSkipped,
    ShowcaseScenario,
    process_events,
)
from datalab.gui.showcase.foreground import bring_to_front

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.gui.showcase.engine import ShowcaseContext

logger = logging.getLogger("datalab.showcase")

#: ID of the interlude, in the list of the scenarios to play
SCENARIO_ID = "web"
#: Public deployment of DataLab Web
DEFAULT_URL = SHOWCASE_WEB_URL
#: Folder of the browser profile, in the DataLab configuration folder
PROFILE_DIRNAME = "showcase-browser"
#: Languages of DataLab Web (other languages of DataLab fall back to English)
WEB_LANGUAGES = ("en", "fr")
#: URL parameters set by the interlude (the other parameters of the URL are kept)
INTERLUDE_PARAMETERS = ("showcase", "loop", "close", "lang", "speed")
#: Delay (s, at normal pace) between the start of the browser and the minimization
#: of the main window, behind the browser window
WINDOW_SWAP_DELAY = 2.0
#: Interval (s) between two checks of the browser process
POLL_INTERVAL = 0.05
#: Interval (s) between two searches of the browser processes using the dedicated
#: profile, once the process started by the interlude has exited
SCAN_INTERVAL = 1.0
#: Delay (s) after the start of the browser, during which the browser processes
#: using the dedicated profile are searched for, even if none is found
STARTUP_GRACE = 3.0
#: Maximum waiting time (s) for the browser processes to end, once asked to
CLOSE_TIMEOUT = 5.0

#: Chromium-based browsers on Windows, by order of preference: executable name,
#: then installation folder relative to the program folders (Microsoft Edge comes
#: last: its enhanced security mode may slow down the Python runtime of DataLab Web)
WINDOWS_BROWSERS = (
    ("chrome.exe", ("Google", "Chrome", "Application")),
    ("msedge.exe", ("Microsoft", "Edge", "Application")),
)
#: Environment variables giving the program folders on Windows
WINDOWS_PROGRAM_FOLDERS = ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")
#: Chromium-based browsers on macOS, by order of preference
MACOS_BROWSERS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
#: Chromium-based browsers on Linux (and other platforms), by order of preference
POSIX_BROWSERS = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
    "microsoft-edge-stable",
)
#: Options of the browser, besides the URL, the profile and the window geometry
BROWSER_OPTIONS = (
    "--no-first-run",
    "--no-default-browser-check",
    "--hide-crash-restore-bubble",
    # No synchronization of the profile with the account of the user (Microsoft
    # Edge signs the profile in, then shows a bubble about it)
    "--disable-sync",
    # DataLab Web keeps its pace even when its window is not in the foreground
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
)
#: Preferences of the dedicated profile: no offer to translate DataLab Web
PROFILE_PREFERENCES = {"translate": {"enabled": False}}


def parse_ids(text: str | None) -> list[str] | None:
    """Parse a comma-separated list of DataLab Web demo IDs

    Args:
        text: Comma-separated IDs (empty, None or ``'all'``: all the demos)

    Returns:
        List of IDs, or None for all the demos
    """
    ids = [part.strip() for part in (text or "").split(",") if part.strip()]
    return None if not ids or ids == ["all"] else ids


def get_ui_language() -> str:
    """Return the language of the user interface of DataLab, for DataLab Web

    Returns:
        Language code (``'en'`` or ``'fr'``)
    """
    translations = getattr(_, "__self__", None)
    info = translations.info() if translations is not None else {}
    language = (info.get("language") or "en").replace("-", "_").split("_")[0]
    language = language.lower()
    return language if language in WEB_LANGUAGES else "en"


def build_web_url(
    base_url: str,
    ids: list[str] | None = None,
    language: str = "en",
    speed: float = 1.0,
) -> str:
    """Return the URL playing one pass of the DataLab Web showcase, then closing
    its window

    The parameters of the base URL are kept (e.g. ``runtime=main``), except those
    set by the interlude: no loop, window closed after one pass, language and pace
    of DataLab. A ``showcase`` parameter of the base URL is kept, unless demo IDs
    are given.

    Args:
        base_url: URL of DataLab Web (e.g. ``https://datalab-platform.com/web/``)
        ids: IDs of the DataLab Web demos to play (None: all the demos)
        language: Language of DataLab Web (e.g. ``'fr'``)
        speed: Delay factor of the demos (0: default pace)

    Returns:
        URL

    Raises:
        ScenarioSkipped: If the URL is not an HTTP(S) or file URL
    """
    parts = urlsplit(base_url.strip())
    if not (
        (parts.scheme in ("http", "https") and parts.netloc) or parts.scheme == "file"
    ):
        raise ScenarioSkipped(f"Invalid DataLab Web URL: {base_url!r}")
    query = parse_qsl(parts.query, keep_blank_values=True)
    showcase = "1"
    for key, value in query:
        if key == "showcase" and value:
            showcase = value
    if ids:
        showcase = ",".join(ids)
    params = [(key, value) for key, value in query if key not in INTERLUDE_PARAMETERS]
    params += [("showcase", showcase), ("close", "1"), ("lang", language)]
    if speed > 0:
        params.append(("speed", f"{speed:g}"))
    return urlunsplit(parts._replace(query=urlencode(params, safe=",")))


def registry_app_path(exe_name: str) -> str | None:
    """Return the path of an application registered in the *App Paths* of Windows

    Args:
        exe_name: Executable name (e.g. ``'msedge.exe'``)

    Returns:
        Path of the executable, or None if not registered (or not on Windows)
    """
    try:
        import winreg  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths" + "\\" + exe_name
    for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(root, key_path) as key:
                value = winreg.QueryValueEx(key, "")[0]
        except OSError:
            continue
        if isinstance(value, str) and value.strip('" '):
            return value.strip('" ')
    return None


def find_browser(platform: str | None = None) -> str | None:
    """Return the path of a Chromium-based browser (Google Chrome, Microsoft Edge
    or Chromium)

    Args:
        platform: Platform (default: ``sys.platform``)

    Returns:
        Path of the browser executable, or None if no browser is found
    """
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        for exe_name, folder in WINDOWS_BROWSERS:
            candidates = [registry_app_path(exe_name), shutil.which(exe_name)]
            for variable in WINDOWS_PROGRAM_FOLDERS:
                base = os.environ.get(variable)
                if base:
                    candidates.append(osp.join(base, *folder, exe_name))
            for path in candidates:
                if path and osp.isfile(path):
                    return path
        return None
    if platform == "darwin":
        for path in MACOS_BROWSERS:
            if osp.isfile(path):
                return path
    for name in POSIX_BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    return None


def browser_command(
    browser: str,
    url: str,
    profile_dir: str,
    fullscreen: bool = False,
    geometry: tuple[int, int, int, int] | None = None,
) -> list[str]:
    """Return the command starting the browser on DataLab Web

    Args:
        browser: Path of the browser executable
        url: URL of DataLab Web, with its showcase parameters
        profile_dir: Folder of the dedicated browser profile
        fullscreen: True to show the browser window in full screen
        geometry: Geometry of the screen showing the browser window (x, y, width,
         height), or None for the default screen

    Returns:
        Command (executable and arguments)
    """
    command = [browser, f"--app={url}", f"--user-data-dir={profile_dir}"]
    command += BROWSER_OPTIONS
    if geometry is not None:
        x, y, width, height = geometry
        command += [f"--window-position={x},{y}", f"--window-size={width},{height}"]
    command.append("--start-fullscreen" if fullscreen else "--start-maximized")
    return command


def get_profile_dir() -> str:
    """Return the folder of the browser profile dedicated to the interlude

    Returns:
        Folder, in the DataLab configuration folder
    """
    return get_config_path(PROFILE_DIRNAME)


def prepare_profile(profile_dir: str) -> None:
    """Create the dedicated browser profile, or update its preferences

    The preferences of :data:`PROFILE_PREFERENCES` are merged into the preferences
    of the default user of the profile (the browser must not run).

    Args:
        profile_dir: Folder of the dedicated browser profile
    """
    path = osp.join(profile_dir, "Default", "Preferences")
    os.makedirs(osp.dirname(path), exist_ok=True)
    preferences: dict = {}
    try:
        with open(path, encoding="utf-8") as fobj:
            preferences = json.load(fobj)
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        logger.warning("Showcase: unreadable browser preferences %s", path)
        return
    if not isinstance(preferences, dict):
        return
    for section, values in PROFILE_PREFERENCES.items():
        if not isinstance(preferences.get(section), dict):
            preferences[section] = {}
        preferences[section].update(values)
    try:
        with open(path, "w", encoding="utf-8") as fobj:
            json.dump(preferences, fobj)
    except OSError:
        logger.warning("Showcase: browser preferences %s not written", path)


def end_processes(processes: list[psutil.Process]) -> None:
    """End processes: terminate them, then kill those still running after a while

    Args:
        processes: Processes
    """
    for process in processes:
        try:
            process.terminate()
        except psutil.Error:
            pass
    _gone, alive = psutil.wait_procs(processes, timeout=CLOSE_TIMEOUT)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass


def uses_profile(cmdline: list[str], profile_dir: str) -> bool:
    """Return True if a command line uses the dedicated browser profile

    Args:
        cmdline: Command line (executable and arguments)
        profile_dir: Folder of the dedicated browser profile

    Returns:
        True if the command line has the ``--user-data-dir`` option of the profile
    """
    target = osp.normcase(osp.normpath(profile_dir))
    for arg in cmdline:
        if arg.startswith("--user-data-dir="):
            folder = arg.split("=", 1)[1].strip('"')
            if folder and osp.normcase(osp.normpath(folder)) == target:
                return True
    return False


def find_profile_processes(
    profile_dir: str, browsers_only: bool = False
) -> list[psutil.Process]:
    """Return the processes running with the dedicated browser profile

    Each process of a Chromium-based browser (the browser itself, its renderers,
    its GPU process...) has the ``--user-data-dir`` option of its profile.

    Args:
        profile_dir: Folder of the dedicated browser profile
        browsers_only: True to leave out the child processes of the browser (which
         have a ``--type`` option)

    Returns:
        Processes
    """
    processes = []
    for process in psutil.process_iter(["cmdline"]):
        cmdline = process.info.get("cmdline") or []
        if process.pid == os.getpid() or not uses_profile(cmdline, profile_dir):
            continue
        if browsers_only and any(arg.startswith("--type=") for arg in cmdline):
            continue
        processes.append(process)
    return processes


def end_profile_processes(profile_dir: str) -> None:
    """End the browser processes left running with the dedicated profile

    A browser started with a profile already in use hands its URL over to the
    running browser, then exits at once: the interlude would not wait for it.

    Args:
        profile_dir: Folder of the dedicated browser profile
    """
    processes = find_profile_processes(profile_dir)
    if processes:
        logger.warning(
            "Showcase: ending %d browser processes left running", len(processes)
        )
        end_processes(processes)


def is_alive(process: psutil.Process) -> bool:
    """Return True if a process runs (and is not a zombie)

    Args:
        process: Process

    Returns:
        True if the process runs
    """
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


class BrowserProcess:
    """Browser started by the interlude, with the dedicated profile

    A browser may start again by itself, in a new process (e.g. Microsoft Edge on
    Windows): the browser runs as long as the process started by the interlude,
    or a browser process with the dedicated profile, runs. The browser processes
    still running when DataLab exits are ended.

    Args:
        command: Command starting the browser
        profile_dir: Folder of the dedicated browser profile
    """

    #: Browsers started, and not yet closed
    running: list[BrowserProcess] = []
    #: True once the browsers are set to be closed when DataLab exits
    exit_handler_registered = False

    def __init__(self, command: list[str], profile_dir: str) -> None:
        self.profile_dir = profile_dir
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        # pylint: disable=consider-using-with
        self.popen = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        self.__started = time.monotonic()
        self.__last_scan = 0.0
        self.__browsers: list[psutil.Process] = []
        self.__closed = False
        if not BrowserProcess.exit_handler_registered:
            atexit.register(BrowserProcess.close_all)
            BrowserProcess.exit_handler_registered = True
        BrowserProcess.running.append(self)

    @property
    def pid(self) -> int:
        """ID of the process started by the interlude"""
        return self.popen.pid

    def is_running(self) -> bool:
        """Return True while the browser runs

        Returns:
            True while the process started by the interlude runs, or a browser
             process with the dedicated profile
        """
        if self.popen.poll() is None:
            return True
        if self.__closed:
            return False
        self.__browsers = [process for process in self.__browsers if is_alive(process)]
        if self.__browsers:
            return True
        now = time.monotonic()
        if now - self.__last_scan < SCAN_INTERVAL:
            return True
        self.__last_scan = now
        self.__browsers = find_profile_processes(self.profile_dir, browsers_only=True)
        return bool(self.__browsers) or now - self.__started < STARTUP_GRACE

    def close(self) -> None:
        """Close the browser window, then end the browser processes"""
        try:
            targets = []
            if self.popen.poll() is None:
                targets.append(self.pid)
            browsers = find_profile_processes(self.profile_dir, browsers_only=True)
            targets += [process.pid for process in browsers]
            try:
                children = psutil.Process(self.pid).children(recursive=True)
            except psutil.Error:
                children = []
            if targets:
                logger.info("Showcase: closing the browser (PID %s)", targets)
                for pid in targets:
                    self.__ask_to_close(pid)
                processes = []
                for pid in targets:
                    try:
                        processes.append(psutil.Process(pid))
                    except psutil.Error:
                        pass
                _gone, alive = psutil.wait_procs(processes, timeout=CLOSE_TIMEOUT)
                if alive:
                    logger.warning("Showcase: the browser did not close, ending it")
            leftovers = [child for child in children if is_alive(child)]
            leftovers += find_profile_processes(self.profile_dir)
            if leftovers:
                end_processes(leftovers)
            if self.popen.poll() is None:
                self.popen.kill()
            self.popen.wait(CLOSE_TIMEOUT)
        finally:
            self.__closed = True
            self.__browsers = []
            if self in BrowserProcess.running:
                BrowserProcess.running.remove(self)

    def __ask_to_close(self, pid: int) -> None:
        """Ask a browser process to close its windows (the browser then exits)

        Args:
            pid: Process ID
        """
        try:
            if sys.platform == "win32":
                # Without the /F option, taskkill asks the windows to close (with
                # the /T option, it fails for the child processes without window)
                subprocess.run(
                    ["taskkill", "/PID", str(pid)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=CLOSE_TIMEOUT,
                    check=False,
                )
            else:
                psutil.Process(pid).terminate()
        except (OSError, subprocess.TimeoutExpired, psutil.Error):
            pass

    @classmethod
    def close_all(cls) -> None:
        """Close the browsers still running (e.g. when DataLab exits)"""
        for browser in list(cls.running):
            try:
                browser.close()
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Showcase: the browser could not be closed")


def wait_for_browser(
    ctx: ShowcaseContext, browser: BrowserProcess, timeout: float
) -> bool:
    """Wait for the browser process to end

    The waiting time is frozen while the operator pauses the showcase (F9). When
    the operator stops the showcase (Shift+F12), the waiting ends at once.

    Args:
        ctx: Showcase context
        browser: Browser process
        timeout: Maximum waiting time (s)

    Returns:
        False after the timeout, True otherwise
    """
    end = time.monotonic() + timeout
    while browser.is_running() and not ctx.stop_requested:
        if ctx.paused:
            remaining = end - time.monotonic()
            ctx.wait_while_paused()
            end = time.monotonic() + remaining
        if time.monotonic() >= end:
            return False
        process_events()
        time.sleep(POLL_INTERVAL)
    return True


def restore_window(win: DLMainWindow, state: QC.Qt.WindowStates) -> None:
    """Show the main window again, in its state before the interlude

    Args:
        win: DataLab main window
        state: Window state before the interlude
    """
    if state & QC.Qt.WindowFullScreen:
        win.showFullScreen()
    elif state & QC.Qt.WindowMaximized:
        win.showMaximized()
    else:
        win.showNormal()
    bring_to_front(win)
    process_events()


def play(
    ctx: ShowcaseContext,
    url: str | None,
    ids: list[str] | None = None,
    timeout: float | None = None,
) -> None:
    """Play the DataLab Web interlude

    Args:
        ctx: Showcase context
        url: URL of DataLab Web (None or empty: the interlude is skipped)
        ids: IDs of the DataLab Web demos to play (default: from the configuration,
         all the demos by default)
        timeout: Maximum duration (s) of the interlude (default: from the
         configuration): the browser is closed afterwards

    Raises:
        ScenarioSkipped: Without URL, with an invalid URL, or without browser
    """
    if not url:
        raise ScenarioSkipped(
            "No DataLab Web URL ('Include DataLab Web' option of the Showcase mode "
            "dialog, or --showcase-web command-line option)"
        )
    if ids is None:
        ids = parse_ids(Conf.showcase_web_ids.get())
    if timeout is None:
        timeout = 60.0 * max(1, Conf.showcase_web_timeout.get())
    web_url = build_web_url(url, ids, get_ui_language(), ctx.speed)
    browser = find_browser()
    if browser is None:
        raise ScenarioSkipped(
            "No Chromium-based browser found (Google Chrome, Microsoft Edge or "
            "Chromium): the DataLab Web interlude needs one"
        )
    win = ctx.win
    state = win.windowState()
    profile_dir = get_profile_dir()
    end_profile_processes(profile_dir)
    prepare_profile(profile_dir)
    screen = win.screen().geometry()
    command = browser_command(
        browser,
        web_url,
        profile_dir,
        fullscreen=bool(state & QC.Qt.WindowFullScreen),
        geometry=(screen.x(), screen.y(), screen.width(), screen.height()),
    )
    ctx.say(_("DataLab Web now takes over, in a browser"), hold=2.5)
    logger.info("Showcase: starting DataLab Web: %s", subprocess.list2cmdline(command))
    process = BrowserProcess(command, profile_dir)
    try:
        # The browser window shows up over the main window, which is then minimized
        ctx.pause(WINDOW_SWAP_DELAY)
        ctx.hide_caption()
        win.showMinimized()
        process_events()
        closed = wait_for_browser(ctx, process, timeout)
    finally:
        process.close()
        restore_window(win, state)
    ctx.checkpoint(
        closed, f"DataLab Web did not close its window within {timeout:.0f} s"
    )
    if not ctx.stop_requested:
        ctx.say(_("Back to DataLab, on the desktop"), hold=2.0)


def create_scenario(
    url: str | None,
    ids: list[str] | None = None,
    timeout: float | None = None,
) -> ShowcaseScenario:
    """Return the scenario of the DataLab Web interlude

    Args:
        url: URL of DataLab Web (None or empty: the interlude is skipped)
        ids: IDs of the DataLab Web demos to play (default: from the configuration)
        timeout: Maximum duration (s) of the interlude (default: from the
         configuration)

    Returns:
        Scenario
    """
    return ShowcaseScenario(
        id=SCENARIO_ID,
        title=_("DataLab Web: the same platform in a browser"),
        subtitle=_("One pass of the DataLab Web showcase, then back to the desktop"),
        run=partial(play, url=url, ids=ids, timeout=timeout),
        duration_hint=300.0,
    )
