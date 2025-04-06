# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal base line selection unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
from guidata.qthelpers import exec_dialog, qt_app_context

from cdl.env import execenv
from cdl.tests.data import create_paracetamol_signal
from cdl.widgets.signalbaseline import SignalBaselineDialog


def test_signal_baseline_selection():
    """Signal baseline selection dialog test"""
    sig = create_paracetamol_signal()
    with qt_app_context():
        dlg = SignalBaselineDialog(sig)
        dlg.resize(640, 480)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        exec_dialog(dlg)
    execenv.print(f"baseline: {dlg.get_baseline()}")
    execenv.print(f"X range: {dlg.get_x_range()}")
    # Check baseline value:
    i0, i1 = np.searchsorted(sig.x, dlg.get_x_range())
    assert dlg.get_baseline() == sig.data[i0:i1].mean()


if __name__ == "__main__":
    test_signal_baseline_selection()
