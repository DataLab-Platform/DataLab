# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.params
import sigima.proc.signal as sips
from qtpy import QtCore as QC
from sigima.tests.data import create_paracetamol_signal

from datalab.env import execenv
from datalab.tests import datalab_test_app_context


def test_history_app():
    """Run history application test scenario"""
    with datalab_test_app_context() as win:
        dock = win.docks[win.historypanel]
        win.addDockWidget(QC.Qt.LeftDockWidgetArea, dock)
        win.resize(int(win.width() * 1.7), win.height())
        win.move(50, 50)
        execenv.print("History application test:")
        panel = win.signalpanel
        sig = create_paracetamol_signal()
        panel.add_object(sig)
        panel.processor.run_feature(sips.derivative)
        panel.processor.run_feature(sips.absolute)
        panel.processor.run_feature(sips.log10)
        param = sigima.params.NormalizeParam.create(method="energy")
        panel.processor.run_feature(sips.normalize, param)
        execenv.print("==> OK")


if __name__ == "__main__":
    test_history_app()
