# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Memory leak test

CodraFT application memory leak test.
"""

import os

import numpy as np
import psutil

from codraft.tests.embedded1_unit import HostWindow
from codraft.utils.qthelpers import qt_app_context
from codraft.utils.vistools import view_curves

SHOW = True  # Show test in GUI-based test launcher


def memory_leak_test(iterations=100):
    """Test for memory leak"""
    with qt_app_context():
        proc = psutil.Process(os.getpid())
        mainview = HostWindow()
        mainview.show()
        memlist = []
        for i in range(iterations):
            mainview.open_codraft()
            mainview.codraft.close()
            # mainview.codraft.destroy()
            # mainview.codraft = None
            # QApplication.processEvents()
            # import time; time.sleep(2)
            # QApplication.processEvents()
            memdata = proc.memory_info().vms / 1024**2
            memlist.append(memdata)
            print(i + 1, ":", memdata, "MB")
        view_curves(
            np.array(memlist),
            title="Memory leak test for CodraFT application",
            ylabel="Memory (MB)",
        )


if __name__ == "__main__":
    memory_leak_test()
