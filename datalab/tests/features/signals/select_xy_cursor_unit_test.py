# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal horizontal or vertical cursor selection unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from typing import Literal

import numpy as np
import pytest
from guidata.qthelpers import exec_dialog, qt_app_context
from sigima.tests.data import create_paracetamol_signal
from sigima.tools.signal.features import find_first_x_at_y_value

from datalab.env import execenv
from datalab.widgets.signalcursor import SignalCursorDialog


@pytest.mark.parametrize("cursor_orientation", ["horizontal", "vertical"])
def test_signal_cursor_selection(
    cursor_orientation: Literal["horizontal", "vertical"],
) -> None:
    """Parametrized signal cursor selection unit test."""
    sig = create_paracetamol_signal()
    with qt_app_context():
        dlg = SignalCursorDialog(signal=sig, cursor_orientation=cursor_orientation)
        dlg.resize(640, 480)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        exec_dialog(dlg)
    x, y = dlg.get_cursor_position()
    if cursor_orientation == "horizontal":
        execenv.print(f"X value: {x}")
        x_sig = find_first_x_at_y_value(sig.x, sig.y, y)
        assert x == x_sig, f"Expected {x_sig}, got {x}"
    else:
        execenv.print(f"Y value: {y}")
        y_sig = sig.y[np.searchsorted(sig.x, x)]
        assert y == y_sig, f"Expected {y_sig}, got {y}"


if __name__ == "__main__":
    test_signal_cursor_selection(cursor_orientation="horizontal")
    test_signal_cursor_selection(cursor_orientation="vertical")
