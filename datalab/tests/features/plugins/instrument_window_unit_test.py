# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Instrument window unit test
---------------------------

Instrument calls run one at a time outside the GUI thread: a slow instrument
does not freeze DataLab, repeated refresh requests are merged, and closing the
window closes the instrument once its running call is done.
"""

# guitest: show

from __future__ import annotations

import threading

import guidata.dataset as gds
import numpy as np
from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW
from sigima.objects import create_signal

from datalab.gui.plugins.instrument import InstrumentWindow
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
        self.running = threading.Event()
        self.release = threading.Event()

    def open(self) -> None:
        self.calls.append("open")

    def preview(self) -> InstrumentFrame:
        self.calls.append("preview")
        self.running.set()
        assert self.release.wait(10.0)
        x = np.linspace(0.0, 1.0, 5)
        return InstrumentFrame((create_signal("Live", x, self.settings.gain * x),))

    def close(self) -> None:
        self.calls.append("close")

    def acquire(self) -> InstrumentAcquisition:
        self.calls.append("acquire")
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


if __name__ == "__main__":
    test_slow_instrument_calls_are_serialized()
    test_close_waits_for_running_call()
