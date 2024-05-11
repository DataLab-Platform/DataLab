# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from qtpy import QtCore as QC

import cdl.obj
import cdl.param
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.data import create_paracetamol_signal


def test_history_app():
    """Run history application test scenario"""
    with cdltest_app_context() as win:
        dock = win.docks[win.historypanel]
        win.addDockWidget(QC.Qt.LeftDockWidgetArea, dock)
        win.resize(int(win.width() * 1.7), win.height())
        win.move(50, 50)
        execenv.print("History application test:")
        panel = win.signalpanel
        sig = create_paracetamol_signal()
        panel.add_object(sig)
        panel.processor.compute_derivative()
        panel.processor.compute_abs()
        panel.processor.compute_log10()
        param = cdl.param.NormalizeParam.create(method="energy")
        panel.processor.compute_normalize(param)
        execenv.print("==> OK")


if __name__ == "__main__":
    test_history_app()
