# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal peak detection dialog test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from guidata.qthelpers import exec_dialog, qt_app_context
from sigima.tests.data import get_test_signal

from cdl.env import execenv
from cdl.widgets.signalpeak import SignalPeakDetectionDialog


def test_peak1d_dialog():
    """Signal peak dialog test"""
    with qt_app_context():
        s = get_test_signal("paracetamol.txt")
        dlg = SignalPeakDetectionDialog(s)
        dlg.resize(640, 300)
        plot = dlg.get_plot()
        plot.set_axis_limits(plot.xBottom, 16, 30)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        exec_dialog(dlg)
    execenv.print("peaks:")
    execenv.pprint(dlg.get_peaks())
    execenv.pprint(dlg.get_min_dist())


if __name__ == "__main__":
    test_peak1d_dialog()
