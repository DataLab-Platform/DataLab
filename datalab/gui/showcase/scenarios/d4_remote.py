# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Remote client of showcase scenario D4
=====================================

A Python client, run in a separate process, drives DataLab through its Web API
(HTTP/JSON), as a Jupyter notebook or DataLab-Kernel would: it sends noisy
acquisitions, asks DataLab for their average, then retrieves the result.

The client only uses the Python standard library and NumPy, which are available
both with the Python package and in the standalone version of DataLab. Its process
runs the Python interpreter of DataLab or, in the standalone version, the DataLab
executable itself, which runs the code given with ``-c`` (see
``datalab/start.pyw``), as macros do.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Any, BinaryIO

from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

logger = logging.getLogger("datalab.showcase")

#: Prefix of the last line written by the client: its result, as JSON
RESULT_PREFIX = "RESULT "

#: Prefix of the temporary files receiving the output of the client
LOG_PREFIX = "datalab-showcase-d4-"

#: Python code of the client
#:
#: The client reads its settings from environment variables: the URL and the
#: access token of the Web API server (``DATALAB_WORKSPACE_URL`` and
#: ``DATALAB_WORKSPACE_TOKEN``, as DataLab-Kernel), the number of acquisitions and
#: the interval between them. Its output goes to the standard output, or to the
#: file ``DATALAB_SHOWCASE_LOG`` without standard output.
CLIENT_CODE = r"""
import io
import json
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
import zipfile


def main(log):
    import numpy as np

    base = os.environ["DATALAB_WORKSPACE_URL"].rstrip("/") + "/api/v1"
    token = os.environ["DATALAB_WORKSPACE_TOKEN"]
    count = int(os.environ["DATALAB_SHOWCASE_COUNT"])
    interval = float(os.environ["DATALAB_SHOWCASE_INTERVAL"])
    # Local server: system proxies are bypassed
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(method, path, body=None, content_type="application/json"):
        req = urllib.request.Request(base + path, data=body, method=method)
        req.add_header("Authorization", "Bearer " + token)
        if body is not None:
            req.add_header("Content-Type", content_type)
        with opener.open(req, timeout=30.0) as response:
            return response.read()

    def object_path(name):
        return "/objects/" + urllib.parse.quote(name, safe="")

    def signal_npz(title, x, y):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, array in (("x.npy", x), ("y.npy", y)):
                data = io.BytesIO()
                np.save(data, array, allow_pickle=False)
                archive.writestr(name, data.getvalue())
            metadata = {
                "type": "signal",
                "title": title,
                "xlabel": "Wavelength",
                "ylabel": "Intensity",
                "xunit": "nm",
                "yunit": "a.u.",
            }
            archive.writestr("metadata.json", json.dumps(metadata))
        return buffer.getvalue()

    def signal_y(data):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return np.load(io.BytesIO(archive.read("y.npy")), allow_pickle=False)

    status = json.loads(request("GET", "/status"))
    log(f"Connected to DataLab {status['version']} at {status['url']}")
    x = np.linspace(400.0, 700.0, 1000)
    lines = ((1.0, 486.1, 6.0), (0.6, 589.0, 4.0), (0.8, 656.3, 8.0))
    clean = sum(a * np.exp(-0.5 * ((x - c) / w) ** 2) for a, c, w in lines)
    rng = np.random.default_rng(2026)
    sigma = 0.15
    names = []
    for index in range(1, count + 1):
        name = f"Acquisition {index:02d}"
        y = clean + rng.normal(0.0, sigma, x.size)
        body = signal_npz(name, x, y)
        path = object_path(name) + "/data?overwrite=true"
        request("PUT", path, body, "application/x-npz")
        names.append(name)
        log(f"PUT {name}: {x.size} points sent")
        time.sleep(interval)
    selection = {"selection": names, "panel": "signal"}
    request("POST", "/select", json.dumps(selection).encode())
    log(f"POST select: {len(names)} acquisitions")
    calc = json.dumps({"name": "average"}).encode()
    reply = json.loads(request("POST", "/calc", calc))
    if not reply["success"] or not reply["result_names"]:
        raise RuntimeError(f"Average not computed: {reply}")
    average = reply["result_names"][0]
    log("POST calc 'average': done")
    y_average = signal_y(request("GET", object_path(average) + "/data"))
    noise = float(np.std(y_average - clean))
    log(f"GET average: {y_average.size} points received")
    log(f"Noise: {sigma:.3f} -> {noise:.3f} ({sigma / noise:.1f} times lower)")
    result = {
        "sent": len(names),
        "average": average,
        "points": int(y_average.size),
        "noise_in": sigma,
        "noise_out": noise,
    }
    log("RESULT " + json.dumps(result))


def run():
    output = sys.stdout
    if output is None:
        output = open(os.environ["DATALAB_SHOWCASE_LOG"], "a", encoding="utf-8")
    else:
        try:
            output.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    def log(text):
        print(text, file=output, flush=True)

    try:
        main(log)
    except Exception:
        traceback.print_exc(file=output)
        output.flush()
        return 1
    return 0


sys.exit(run())
"""


def is_serving(url: str, timeout: float = 1.0) -> bool:
    """Return True if a Web API server answers at an URL

    Args:
        url: Web API server URL (e.g. ``'http://127.0.0.1:18080'``)
        timeout: Timeout (s) of the request

    Returns:
        True if the status of the server is available
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url.rstrip("/") + "/api/v1/status", timeout=timeout) as resp:
            return resp.status == 200
    except (OSError, ValueError):
        return False


class ClientProcess(QC.QObject):
    """Process of the client, driving DataLab through its Web API

    The output of the process is written to a temporary file, read while the
    process runs (the standalone version of DataLab has no console): each new line
    is emitted with :attr:`SIG_OUTPUT`.

    Args:
        parent: Parent object (e.g. the main window)
        url: Web API server URL
        token: Access token of the Web API server
        count: Number of acquisitions sent by the client
        interval: Interval (s) between two acquisitions
    """

    #: Emitted for each new line written by the client
    SIG_OUTPUT = QC.Signal(str)

    #: Interval (ms) between two readings of the output of the client
    POLL_INTERVAL_MS = 100

    def __init__(
        self, parent: QC.QObject, url: str, token: str, count: int, interval: float
    ) -> None:
        super().__init__(parent)
        self.url = url
        self.token = token
        self.count = count
        self.interval = interval
        #: Lines written by the client so far
        self.lines: list[str] = []
        self.__process: subprocess.Popen | None = None
        self.__log_path: str | None = None
        self.__reader: BinaryIO | None = None
        self.__partial = b""
        self.__timer = QC.QTimer(self)
        self.__timer.setInterval(self.POLL_INTERVAL_MS)
        self.__timer.timeout.connect(self.poll)

    @staticmethod
    def command() -> list[str]:
        """Return the command line running the client

        With the Python package, the client runs with the Python interpreter of
        DataLab. In the standalone version, ``sys.executable`` is the DataLab
        executable, which runs the code given with ``-c`` without starting DataLab.

        Returns:
            Command line
        """
        return [sys.executable, "-c", CLIENT_CODE]

    @property
    def finished(self) -> bool:
        """True once the client process has exited"""
        return self.__process is not None and self.__process.poll() is not None

    @property
    def returncode(self) -> int | None:
        """Exit code of the client process (None while running)"""
        return None if self.__process is None else self.__process.poll()

    def start(self) -> None:
        """Start the client process"""
        handle, self.__log_path = tempfile.mkstemp(prefix=LOG_PREFIX, suffix=".log")
        env = dict(
            os.environ,
            DATALAB_WORKSPACE_URL=self.url,
            DATALAB_WORKSPACE_TOKEN=self.token,
            DATALAB_SHOWCASE_COUNT=str(self.count),
            DATALAB_SHOWCASE_INTERVAL=str(self.interval),
            DATALAB_SHOWCASE_LOG=self.__log_path,
        )
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        with os.fdopen(handle, "wb") as output:
            self.__process = subprocess.Popen(  # pylint: disable=consider-using-with
                self.command(),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
                **kwargs,
            )
        # pylint: disable-next=consider-using-with
        self.__reader = open(self.__log_path, "rb")
        logger.info("Showcase client process started (PID %d)", self.__process.pid)
        self.__timer.start()

    def poll(self) -> None:
        """Read the new output of the client, and emit its complete lines"""
        if self.__reader is None:
            return
        data = self.__partial + self.__reader.read()
        *complete, self.__partial = data.split(b"\n")
        if self.finished and self.__partial:
            complete.append(self.__partial)
            self.__partial = b""
        for raw in complete:
            line = raw.decode("utf-8", errors="replace").rstrip("\r")
            self.lines.append(line)
            self.SIG_OUTPUT.emit(line)

    def has_line(self, prefix: str) -> bool:
        """Return True if the client has written a line starting with a prefix

        Args:
            prefix: Line prefix

        Returns:
            True if such a line has been written
        """
        return any(line.startswith(prefix) for line in self.lines)

    def result(self) -> dict[str, Any] | None:
        """Return the result written by the client, if any

        Returns:
            Result (number of acquisitions sent, title of their average, number of
             points and noise level of the average retrieved...), or None
        """
        for line in reversed(self.lines):
            if line.startswith(RESULT_PREFIX):
                try:
                    return json.loads(line[len(RESULT_PREFIX) :])
                except ValueError:
                    return None
        return None

    def stop(self) -> None:
        """Stop the client process (if still running) and remove its output file"""
        self.__timer.stop()
        process = self.__process
        if process is not None and process.poll() is None:
            logger.warning("Showcase client process killed (PID %d)", process.pid)
            process.kill()
        if process is not None:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                logger.error("Showcase client process does not exit")
        if self.__reader is not None:
            self.poll()
            self.__reader.close()
            self.__reader = None
        if self.__log_path is not None:
            for _attempt in range(20):
                try:
                    os.remove(self.__log_path)
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                logger.warning("Cannot remove %s", self.__log_path)
            self.__log_path = None


class ClientConsole(QW.QFrame):
    """Console showing the output of the client, over the main window

    The console is a child widget of the main window, over the lower part of the
    signal and image panels (the object list and the plots stay visible).

    Args:
        win: Main window
        title: Title of the console
    """

    def __init__(self, win: QW.QMainWindow, title: str) -> None:
        super().__init__(win)
        self.win = win
        self.setObjectName("showcase_client_console")
        self.setAttribute(QC.Qt.WA_StyledBackground)
        self.setStyleSheet(
            "#showcase_client_console {background: rgb(16, 22, 34);"
            " border: 2px solid rgb(130, 185, 255); border-radius: 12px;}"
        )
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        label = QW.QLabel(title, self)
        label.setStyleSheet("color: rgb(130, 185, 255); font-weight: bold;")
        self.editor = QW.QPlainTextEdit(self)
        self.editor.setReadOnly(True)
        self.editor.setFocusPolicy(QC.Qt.NoFocus)
        self.editor.setFrameShape(QW.QFrame.NoFrame)
        self.editor.setStyleSheet(
            "QPlainTextEdit {background: transparent; color: rgb(225, 235, 245);}"
        )
        font = QG.QFontDatabase.systemFont(QG.QFontDatabase.FixedFont)
        font.setPointSizeF(max(9.0, min(13.0, win.height() * 0.011)))
        self.editor.setFont(font)
        layout.addWidget(label)
        layout.addWidget(self.editor)
        win.installEventFilter(self)

    def eventFilter(  # pylint: disable=invalid-name
        self, obj: QC.QObject, event: QC.QEvent
    ) -> bool:
        """Follow the size of the main window

        Args:
            obj: Watched object
            event: Event

        Returns:
            False: events are never filtered out
        """
        if obj is self.win and event.type() == QC.QEvent.Resize and self.isVisible():
            self.relayout()
        return False

    def relayout(self) -> None:
        """Place the console over the lower part of the data panels"""
        win = self.win
        tabwidget = win.tabwidget
        area = QC.QRect(tabwidget.mapTo(win, QC.QPoint(0, 0)), tabwidget.size())
        if area.width() < 360:
            area = win.rect()
        margin = 8
        top = area.top() + int(area.height() * 0.42)
        bottom = min(area.bottom() - margin, int(win.height() * 0.72))
        bottom = max(bottom, top + 160)
        self.setGeometry(
            area.left() + margin, top, area.width() - 2 * margin, bottom - top
        )

    def popup(self) -> None:
        """Show the console over the other widgets of the main window"""
        self.relayout()
        self.show()
        self.raise_()

    def append(self, line: str) -> None:
        """Append a line to the console

        Args:
            line: Line
        """
        self.editor.appendPlainText(line)
        self.editor.verticalScrollBar().setValue(
            self.editor.verticalScrollBar().maximum()
        )

    def dispose(self) -> None:
        """Hide the console, then delete it"""
        self.win.removeEventFilter(self)
        self.hide()
        self.deleteLater()
