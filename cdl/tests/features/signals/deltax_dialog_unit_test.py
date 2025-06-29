# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal delta x dialog unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from guidata.qthelpers import exec_dialog, qt_app_context

from cdl.widgets.signaldeltax import SignalDeltaXDialog
from sigima_.algorithms.signal.pulse import full_width_at_y
from sigima_.tests.data import create_paracetamol_signal


def test_signal_delta_x_dialog():
    """Test the SignalDeltaXDialog widget."""
    sig = create_paracetamol_signal()
    with qt_app_context():
        dlg = SignalDeltaXDialog(signal=sig)
        dlg.resize(640, 480)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        exec_dialog(dlg)
    y = dlg.get_y_value()
    x0, y0, x1, y1 = dlg.get_coords()
    exp_x0, exp_y0, exp_x1, exp_y1 = full_width_at_y((sig.x, sig.y), y)
    assert (x0, y0, x1, y1) == (exp_x0, exp_y0, exp_x1, exp_y1), (
        f"Expected: {(exp_x0, exp_y0, exp_x1, exp_y1)} but got: {(x0, y0, x1, y1)}"
    )


if __name__ == "__main__":
    test_signal_delta_x_dialog()
