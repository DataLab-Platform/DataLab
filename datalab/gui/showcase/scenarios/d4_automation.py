# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenario D4: automation and openness
=============================================

Macros and remote control:

1. A macro is written in the Macro Panel, then run: in its own Python process, it
   creates chirp signals with NumPy, and computes their spectrum with DataLab.
2. The Web API server (HTTP/JSON) is started from the **File > Web API** menu.
3. A separate Python process, as a Jupyter notebook would, drives DataLab through
   the Web API: it sends noisy acquisitions, one by one, asks DataLab for their
   average, then retrieves the result (see
   :mod:`datalab.gui.showcase.scenarios.d4_remote`).
4. The Web API server is stopped.

The scenario leaves no trace behind: the macro is removed, along with its entry
in the macro recovery cache (otherwise, DataLab would offer to recover it at next
startup), the client process is stopped, its output file is removed, the Web API
server is stopped and the connection file of another Web API server, if any, is
restored. The remote control server (XML-RPC), used by macros, is started for the
macro if it is disabled, then stopped.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.showcase.engine import ScenarioSkipped, ShowcaseScenario
from datalab.gui.showcase.scenarios.d4_remote import (
    RESULT_PREFIX,
    ClientConsole,
    ClientProcess,
    is_serving,
)
from datalab.webapi import webapi_unavailable_message
from datalab.widgets.codecompleter import PythonCompleter

if TYPE_CHECKING:
    from datalab.gui.macroeditor import Macro
    from datalab.gui.main import DLMainWindow
    from datalab.gui.showcase.engine import ShowcaseContext

logger = logging.getLogger("datalab.showcase")

#: Title of the macro written by the scenario
MACRO_TITLE = "chirp_analysis"

#: Code of the macro, typed in the macro editor
#:
#: The spectrum is computed with ``call_method("calc", ..., edit=False)``: unlike
#: ``proxy.calc``, it shows no parameter dialog (whose nested event loop could not
#: run once a previous Qt event loop has quit, e.g. in the test suite).
MACRO_CODE = """import numpy as np
from datalab.control.proxy import RemoteProxy

proxy = RemoteProxy()
t = np.linspace(0, 1, 2000)
for f in (100, 200):
    y = np.sin(np.pi * f * t**2)  # Chirp: 0 to f Hz
    proxy.add_signal(f"Chirp {f} Hz", t, y, "s", "V")
    proxy.call_method("calc", "magnitude_spectrum", edit=False)
    print(f"Chirp {f} Hz: spectrum computed")
print("Done!")
"""

#: Number of signals created by the macro (chirps and spectra)
MACRO_SIGNALS = 4

#: Number of acquisitions sent by the remote client
ACQUISITION_COUNT = 8

#: Maximum duration (s) of the macro
MACRO_TIMEOUT = 90.0

#: Maximum duration (s) of the remote client, besides the acquisition intervals
CLIENT_TIMEOUT = 90.0

#: Maximum duration (s) of the startup of a server
SERVER_TIMEOUT = 15.0


def find_action(win: DLMainWindow, text: str) -> QW.QAction | None:
    """Return the action of the main window with a text

    Args:
        win: Main window
        text: Action text

    Returns:
        Action, or None if not found
    """
    for action in win.findChildren(QW.QAction):
        if action.text() == text:
            return action
    return None


def mask_token(token: str) -> str:
    """Return an access token, masked except its first characters

    Args:
        token: Access token

    Returns:
        Masked token (e.g. ``'aB3x…'``)
    """
    return token[:4] + "…"


# ------ Macro ---------------------------------------------------------------------
@contextmanager
def remote_control_server(ctx: ShowcaseContext) -> Generator[None, None, None]:
    """Run the remote control server (XML-RPC), used by macros, within the context

    If the server is disabled (see the settings), it is started for the context,
    then stopped.

    Args:
        ctx: Showcase context

    Yields:
        None
    """
    server = ctx.win.remote_server
    if server.isRunning():
        yield
        return
    logger.info("Showcase: starting the remote control server for the macro")
    server.start()
    try:
        ready = ctx.wait_until(
            lambda: (
                server.server is not None
                and server.port is not None
                and execenv.xmlrpcport == server.port
            ),
            SERVER_TIMEOUT,
        )
        ctx.checkpoint(ready, "Remote control server not started")
        yield
    finally:
        server.shutdown_server()
        server.wait(10000)
        logger.info("Showcase: remote control server stopped")


def remove_macro(ctx: ShowcaseContext, macro: Macro) -> None:
    """Remove a macro, without confirmation, and its entry in the recovery cache

    Args:
        ctx: Showcase context
        macro: Macro
    """
    panel = ctx.win.macropanel
    if macro.is_running():
        macro.kill()
        ctx.wait_until(lambda: not macro.is_running(), 10.0)
    number = panel.get_number_from_macro(macro)
    if number is not None:
        # The demo mode is enabled: the macro is removed without confirmation
        panel.remove_macro(number)
    # The macro was saved in the recovery cache while typed, then before running
    macro.clear_autosave()


@contextmanager
def no_completion(editor: QW.QWidget) -> Generator[None, None, None]:
    """Disable the code completion of an editor within the context

    The completion popup would show up whenever the typing pauses, e.g. on a
    loaded computer.

    Args:
        editor: Code editor

    Yields:
        None
    """
    timers = [
        timer
        for completer in editor.findChildren(PythonCompleter)
        for timer in completer.findChildren(QC.QTimer)
    ]
    for timer in timers:
        timer.blockSignals(True)
    try:
        yield
    finally:
        for timer in timers:
            timer.stop()
            timer.blockSignals(False)


def play_macro(ctx: ShowcaseContext) -> None:
    """Write a macro, then run it

    Args:
        ctx: Showcase context
    """
    win = ctx.win
    panel = win.macropanel
    win.set_current_panel("macro")
    ctx.say(_("Macros automate DataLab with Python scripts"), hold=2.5)
    ctx.point_action(panel.new_action)
    macro = panel.add_blank_macro()
    try:
        macro.title = MACRO_TITLE
        with ctx.spotlight(panel):
            ctx.say(
                _(
                    "A new macro: a few lines of Python, with NumPy and the "
                    "DataLab proxy"
                ),
                hold=1.0,
            )
            with no_completion(macro.editor):
                ctx.type_text(macro.editor, MACRO_CODE)
            ctx.say(
                _(
                    "The macro creates two chirp signals with NumPy, then asks "
                    "DataLab for their spectrum"
                ),
                hold=3.5,
            )
        with remote_control_server(ctx):
            finished: list[bool] = []
            macro.FINISHED.connect(lambda: finished.append(True))
            count = len(win.signalpanel)
            # Only the output of the macro is shown in the console
            panel.console.clear()
            ctx.trigger_action(panel.run_action)
            ctx.say(
                _(
                    "The macro runs in its own Python process, and drives DataLab "
                    "remotely"
                ),
                hold=2.0,
            )
            done = ctx.wait_until(lambda: bool(finished), MACRO_TIMEOUT)
            output = panel.console.toPlainText()[-1500:]
            ctx.checkpoint(done, "Macro still running after the timeout")
            ctx.checkpoint(macro.get_exit_code() == 0, "Macro failed:\n" + output)
        created = len(win.signalpanel) - count
        ctx.checkpoint(
            created == MACRO_SIGNALS,
            f"Macro created {created} signals instead of {MACRO_SIGNALS}",
        )
        # Signals created by the macro raise the plots: back to the console
        win.set_current_panel("macro")
        with ctx.spotlight(panel.console):
            ctx.say(_("Its output is shown in the console of the Macro Panel"), 2.0)
            # The caption would hide the last lines of the console
            ctx.hide_caption()
            ctx.pause(2.5)
        win.set_current_panel("signal")
        # Spectra of the chirps, overlaid
        objs = list(win.signalpanel.objmodel)[count:]
        spectra = [obj for obj in objs if not obj.title.startswith("Chirp")]
        win.signalpanel.objview.select_objects(spectra)
        ctx.say(
            _("Signals and spectra created by the macro, ready for further analysis"),
            hold=3.5,
        )
    finally:
        remove_macro(ctx, macro)
    ctx.checkpoint(macro not in list(panel), "Macro not removed")


# ------ Web API -------------------------------------------------------------------
@contextmanager
def keep_connection_file() -> Generator[None, None, None]:
    """Restore the connection file of the Web API server, after the context

    The Web API server writes its connection file when started, and removes it
    when stopped: the file of another DataLab instance (e.g. the one of the
    operator) is restored.

    Yields:
        None
    """
    # pylint: disable-next=import-outside-toplevel
    from datalab.webapi.controller import get_connection_file_path

    try:
        path = get_connection_file_path()
        previous = path.read_bytes() if path.is_file() else None
    except OSError:
        path, previous = None, None
    try:
        yield
    finally:
        if path is not None and previous is not None and not path.exists():
            try:
                path.write_bytes(previous)
            except OSError:
                logger.warning("Cannot restore the Web API connection file")


def play_client(ctx: ShowcaseContext, url: str, token: str) -> None:
    """Drive DataLab from a separate client process, through the Web API

    Args:
        ctx: Showcase context
        url: Web API server URL
        token: Access token
    """
    win = ctx.win
    win.set_current_panel("signal")
    ctx.say(
        _(
            "A separate Python process, like a Jupyter notebook, now drives "
            "DataLab through this API"
        ),
        hold=3.0,
    )
    # One acquisition per second at normal pace, without delay in check mode
    client = ClientProcess(win, url, token, ACQUISITION_COUNT, ctx.speed)
    console = None
    if not ctx.check:
        console = ClientConsole(win, _("Python client (separate process)"))

        def show_line(line: str) -> None:
            if not line.startswith(RESULT_PREFIX):
                console.append(line)

        client.SIG_OUTPUT.connect(show_line)
        console.popup()
    try:
        client.start()
        timeout = CLIENT_TIMEOUT + ACQUISITION_COUNT * ctx.speed
        ctx.wait_until(lambda: client.has_line("PUT ") or client.finished, timeout)
        ctx.say(
            _("The client sends %d noisy acquisitions, one per second...")
            % ACQUISITION_COUNT,
            hold=0.5,
        )
        ctx.wait_until(lambda: client.has_line("POST ") or client.finished, timeout)
        ctx.say(
            _("...then asks DataLab for their average, and retrieves the result"),
            hold=1.0,
        )
        done = ctx.wait_until(lambda: client.finished, timeout)
        client.poll()
        output = "\n".join(client.lines[-20:])
        ctx.checkpoint(done, "Client still running after the timeout:\n" + output)
        ctx.checkpoint(client.returncode == 0, "Client failed:\n" + output)
        result = client.result()
        ctx.checkpoint(result is not None, "No result from the client:\n" + output)
        objs = list(win.signalpanel.objmodel)
        received = [obj for obj in objs if obj.title.startswith("Acquisition ")]
        ctx.checkpoint(
            result["sent"] == len(received) == ACQUISITION_COUNT,
            f"{len(received)} acquisitions received instead of {ACQUISITION_COUNT}",
        )
        average = [obj for obj in objs if obj.title == result["average"]]
        ctx.checkpoint(len(average) == 1, "Average not found in DataLab")
        ctx.checkpoint(
            result["points"] == received[0].y.size
            and result["noise_out"] < result["noise_in"] / 2.0,
            f"Unexpected average retrieved by the client: {result}",
        )
        # The first acquisition and the average, overlaid
        win.signalpanel.objview.select_objects([received[0], average[0]])
        reduction = round(100.0 * (1.0 - result["noise_out"] / result["noise_in"]))
        ctx.say(
            _("Average of %d acquisitions, computed by DataLab: noise reduced by %d%%")
            % (ACQUISITION_COUNT, reduction),
            hold=5.0,
        )
    finally:
        client.stop()
        client.deleteLater()
        if console is not None:
            console.dispose()


def play_remote_control(ctx: ShowcaseContext) -> None:
    """Start the Web API server, drive DataLab from a client, then stop the server

    A Web API server already running (e.g. started by the operator) is used, and
    left running.

    Args:
        ctx: Showcase context
    """
    win = ctx.win
    ctx.say(
        _("DataLab can also be controlled remotely, from Jupyter or any application"),
        hold=3.0,
    )
    status = win.get_webapi_status()
    started = not status["running"]
    with keep_connection_file():
        try:
            if started:
                start_action = find_action(win, _("Start Web API Server"))
                if start_action is not None:
                    ctx.point_action(start_action)
                info = win.start_webapi_server()
                url, token = info["url"], info["token"]
            else:
                url, token = status["url"], status["token"]
            serving = ctx.wait_until(lambda: is_serving(url), SERVER_TIMEOUT)
            ctx.checkpoint(serving, f"Web API server not answering at {url}")
            ctx.checkpoint(win.get_webapi_status()["running"], "Web API not running")
            with ctx.spotlight(win.webapistatus):
                ctx.say(
                    _(
                        "Web API server listening at %s, protected by an access "
                        "token (%s)"
                    )
                    % (url, mask_token(token)),
                    hold=4.5,
                )
            play_client(ctx, url, token)
            if started:
                stop_action = find_action(win, _("Stop Web API Server"))
                if stop_action is not None:
                    ctx.point_action(stop_action)
                win.stop_webapi_server()
                ctx.checkpoint(
                    not win.get_webapi_status()["running"], "Web API still running"
                )
        finally:
            if started and win.get_webapi_status()["running"]:
                win.stop_webapi_server()
    ctx.say(_("Jupyter, scripts, any HTTP client can drive DataLab"), hold=5.0)


def run(ctx: ShowcaseContext) -> None:
    """Play the scenario

    Args:
        ctx: Showcase context

    Raises:
        ScenarioSkipped: If the Web API is not available
    """
    if not ctx.win.get_webapi_status()["available"]:
        raise ScenarioSkipped("Scenario d4: " + webapi_unavailable_message())
    play_macro(ctx)
    play_remote_control(ctx)


SCENARIO = ShowcaseScenario(
    id="d4",
    title=_("Automation & openness: macros and remote control"),
    subtitle=_("Python macros, Jupyter notebooks and remote control"),
    run=run,
    requires=("fastapi", "uvicorn", "pydantic"),
    duration_hint=105.0,
)
