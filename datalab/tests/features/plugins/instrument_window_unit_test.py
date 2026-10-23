# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Instrument window unit test
---------------------------

Instrument calls run one at a time outside the GUI thread: a slow instrument
does not freeze DataLab, repeated refresh requests are merged, and closing the
window closes the instrument once its running call is done, even if the window
was never shown. A window opened again with the worker of a closed one calls
the instrument once it is closed. Live mode switched off during an acquisition
is not resumed after it. When opening the instrument fails, the reason stays in
view and the instrument is opened again before the next call. When DataLab
closes, the instruments are waited for within a common time limit.
"""

# guitest: show

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import guidata.dataset as gds
import numpy as np
from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW
from sigima.objects import create_signal

from datalab.gui.plugins.instrument import (
    InstrumentWindow,
    SerialWorker,
    wait_instruments_closed,
)
from datalab.plugins.instruments import (
    InstrumentAcquisition,
    InstrumentFrame,
    PluginInstrument,
)


class SlowSettings(gds.DataSet):
    """Settings of the slow instrument."""

    gain = gds.FloatItem("Gain", default=1.0)


class SlowInstrument(PluginInstrument):
    """Instrument whose previews wait for the test to release them."""

    def __init__(self) -> None:
        super().__init__(SlowSettings())
        self.calls: list[str] = []
        self.threads: set[int] = set()
        self.running = threading.Event()
        self.release = threading.Event()

    def _record(self, method: str) -> None:
        """Record a call and the thread running it."""
        self.calls.append(method)
        self.threads.add(threading.get_ident())

    def open(self) -> None:
        self._record("open")

    def preview(self) -> InstrumentFrame:
        self._record("preview")
        self.running.set()
        assert self.release.wait(10.0)
        x = np.linspace(0.0, 1.0, 5)
        return InstrumentFrame((create_signal("Live", x, self.settings.gain * x),))

    def close(self) -> None:
        self._record("close")

    def acquire(self) -> InstrumentAcquisition:
        self._record("acquire")
        return InstrumentAcquisition("Slow", self.preview().objects)


def test_slow_instrument_calls_are_serialized() -> None:
    """Refresh requests made during a slow preview are merged into one."""
    with qt_app_context():
        main = QW.QWidget()
        instrument = SlowInstrument()
        window = InstrumentWindow(main, instrument, "Slow instrument")
        window.show()
        while not instrument.running.is_set():
            QW.QApplication.processEvents()
        assert not window.is_idle()
        for _index in range(3):
            window.refresh()
        assert instrument.calls == ["open", "preview"]
        instrument.release.set()
        assert window.wait_until_idle()
        assert instrument.calls == ["open", "preview", "preview"]
        assert len(window.items) == 1
        window.close()
        assert window.wait_closed()
        assert instrument.calls[-1] == "close"
        main.deleteLater()


def test_close_waits_for_running_call() -> None:
    """The instrument is closed after its running call, and nothing follows."""
    with qt_app_context():
        main = QW.QWidget()
        instrument = SlowInstrument()
        window = InstrumentWindow(main, instrument, "Slow instrument")
        window.show()
        while not instrument.running.is_set():
            QW.QApplication.processEvents()
        window.refresh()
        window.close()
        assert not window.wait_closed(0.2)
        instrument.release.set()
        assert window.wait_closed()
        assert instrument.calls == ["open", "preview", "close"]
        main.deleteLater()


def test_reopened_window_waits_for_closing() -> None:
    """A window opened again while the closed one still runs a call (sharing its
    worker, as DataLab does) calls the instrument once it is closed, in the same
    thread."""
    with qt_app_context():
        main = QW.QWidget()
        instrument = SlowInstrument()
        worker = SerialWorker()
        first = InstrumentWindow(main, instrument, "Slow instrument", worker)
        first.show()
        while not instrument.running.is_set():
            QW.QApplication.processEvents()
        first.close()
        second = InstrumentWindow(main, instrument, "Slow instrument", worker)
        second.show()
        QW.QApplication.processEvents()
        assert instrument.calls == ["open", "preview"]
        instrument.release.set()
        assert first.wait_closed()
        assert second.wait_until_idle()
        assert instrument.calls == ["open", "preview", "close", "open", "preview"]
        assert len(instrument.threads) == 1
        second.close()
        assert second.wait_closed()
        assert instrument.calls[-1] == "close"
        worker.shutdown()
        main.deleteLater()


def test_live_mode_after_acquisition() -> None:
    """Live mode is resumed after an acquisition, unless it was switched off
    while the acquisition was running."""
    with qt_app_context():
        main = QW.QWidget()
        instrument = SlowInstrument()
        instrument.release.set()
        window = InstrumentWindow(main, instrument, "Slow instrument")
        window.show()
        assert window.wait_until_idle()
        live_timer = window._live_timer  # pylint: disable=protected-access
        for keep_live in (False, True):
            window.live_button.setChecked(True)
            instrument.release.clear()
            instrument.running.clear()
            window.acquire()
            assert instrument.running.wait(10.0)
            assert not live_timer.isActive()
            window.live_button.setChecked(keep_live)
            instrument.release.set()
            assert window.wait_until_idle()
            assert live_timer.isActive() == keep_live
        window.live_button.setChecked(False)
        window.close()
        assert window.wait_closed()
        assert instrument.calls.count("acquire") == 2
        main.deleteLater()


def test_hidden_window_closes_instrument() -> None:
    """Closing a window that was never shown closes its instrument."""
    with qt_app_context():
        main = QW.QWidget()
        instrument = SlowInstrument()
        instrument.release.set()
        window = InstrumentWindow(main, instrument, "Slow instrument")
        window.close()
        assert window.wait_closed()
        assert instrument.calls[0] == "open" and instrument.calls[-1] == "close"
        main.deleteLater()


class FailingInstrument(SlowInstrument):
    """Instrument that cannot be opened until the test allows it."""

    def __init__(self) -> None:
        super().__init__()
        self.fail = True
        self.release.set()

    def open(self) -> None:
        self._record("open")
        if self.fail:
            raise ValueError("Device unreachable")


def test_open_failure_is_shown_and_retried() -> None:
    """When opening fails, the reason stays in view and the instrument is not
    called: the next request opens it again."""
    with qt_app_context():
        main = QW.QWidget()
        instrument = FailingInstrument()
        window = InstrumentWindow(main, instrument, "Failing instrument")
        window.show()
        assert window.wait_until_idle()
        assert instrument.calls == ["open"]
        assert window.status_label.text() == "Device unreachable"
        window.refresh()
        assert window.wait_until_idle()
        assert instrument.calls == ["open", "open"]
        assert window.status_label.text() == "Device unreachable"
        instrument.fail = False
        window.refresh()
        assert window.wait_until_idle()
        assert instrument.calls == ["open", "open", "open", "preview"]
        assert window.status_label.text() == ""
        assert len(window.items) == 1
        window.close()
        assert window.wait_closed()
        assert instrument.calls[-1] == "close"
        main.deleteLater()


def test_instruments_close_within_common_time_limit() -> None:
    """Instruments are waited for together: their time limits do not add up."""
    closed: Future = Future()
    closed.set_result(None)
    closings = [("Stuck 1", Future()), ("Closed", closed), ("Stuck 2", Future())]
    start = time.perf_counter()
    assert wait_instruments_closed(closings, 0.5) == ["Stuck 1", "Stuck 2"]
    assert time.perf_counter() - start < 0.9


if __name__ == "__main__":
    test_slow_instrument_calls_are_serialized()
    test_close_waits_for_running_call()
    test_reopened_window_waits_for_closing()
    test_live_mode_after_acquisition()
    test_hidden_window_closes_instrument()
    test_open_failure_is_shown_and_retried()
    test_instruments_close_within_common_time_limit()
