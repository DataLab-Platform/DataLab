# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Signal peak detection test

Testing peak detection dialog box.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from guidata.qthelpers import exec_dialog, qt_app_context

from cdlapp.env import execenv
from cdlapp.tests.data import get_test_signal
from cdlapp.widgets.signalpeakdialog import SignalPeakDetectionDialog


def test():
    """Signal peak dialog test"""
    with qt_app_context():
        s = get_test_signal("paracetamol.txt")
        dlg = SignalPeakDetectionDialog()
        dlg.resize(640, 300)
        dlg.setup_data(s.x, s.y)
        plot = dlg.get_plot()
        plot.set_axis_limits(plot.xBottom, 16, 30)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        exec_dialog(dlg)
    execenv.print("peaks:")
    execenv.pprint(dlg.get_peaks())
    execenv.pprint(dlg.get_min_dist())


if __name__ == "__main__":
    test()
