# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Baseline dialog test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import numpy as np
import sigima.computation.signal as sigima_signal
import sigima.objects
from guidata.qthelpers import exec_dialog, qt_app_context
from sigima.tests.data import create_paracetamol_signal
from sigima.tests.vistools import view_curves

from datalab.env import execenv
from datalab.widgets.signalbaseline import SignalBaselineDialog


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


def test_signal_baseline_dialog() -> None:
    """Test the signal baseline dialog for offset correction."""
    with qt_app_context():
        s1 = create_paracetamol_signal()
        dlg = SignalBaselineDialog(s1)
        if exec_dialog(dlg):
            param = sigima.objects.ROI1DParam()
            param.xmin, param.xmax = dlg.get_x_range()
            s2 = sigima_signal.offset_correction(s1, param)
            view_curves([s1, s2], title="Signal offset correction")


if __name__ == "__main__":
    test_signal_baseline_selection()
    test_signal_baseline_dialog()
