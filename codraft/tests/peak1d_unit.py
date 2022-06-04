# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Signal peak detection test

Testing peak detection dialog box.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from pprint import pprint

import numpy as np

from codraft.utils.qthelpers import exec_dialog, qt_app_context
from codraft.utils.tests import get_test_fnames
from codraft.widgets.signalpeakdialog import SignalPeakDetectionDialog

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Signal peak dialog test"""
    with qt_app_context():
        data = np.loadtxt(get_test_fnames("paracetamol.txt")[0], delimiter=",")
        x, y = data.T
        dlg = SignalPeakDetectionDialog()
        dlg.resize(640, 300)
        dlg.setup_data(x, y)
        plot = dlg.get_plot()
        plot.set_axis_limits(plot.xBottom, 16, 30)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        exec_dialog(dlg)
    print("peaks:")
    pprint(dlg.get_peaks())
    pprint(dlg.get_min_dist())


if __name__ == "__main__":
    test()
