# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal base line selection unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from guidata.qthelpers import exec_dialog, qt_app_context

from cdl.env import execenv
from cdl.tests.data import get_test_signal
from cdl.widgets.signalbaseline import SignalBaselineDialog


def test_signal_baseline_selection():
    """Signal baseline selection dialog test"""
    with qt_app_context():
        s = get_test_signal("paracetamol.txt")
        dlg = SignalBaselineDialog(s)
        dlg.resize(640, 480)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        exec_dialog(dlg)
    execenv.print(f"baseline: {dlg.get_baseline()}")
    execenv.print(f"index range: {dlg.get_index_range()}")


if __name__ == "__main__":
    test_signal_baseline_selection()
