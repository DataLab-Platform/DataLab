# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Memory leak test

DataLab application memory leak test.
"""

# guitest: skip

import os

import numpy as np
import psutil
from guidata.qthelpers import qt_app_context

from cdl.env import execenv
from cdl.tests.features.control.embedded1_unit_test import HostWindow
from cdl.utils.vistools import view_curves


def memory_leak_test(iterations=100):
    """Test for memory leak"""
    with qt_app_context():
        proc = psutil.Process(os.getpid())
        mainview = HostWindow()
        mainview.show()
        memlist = []
        for i in range(iterations):
            mainview.init_cdl()
            mainview.close_cdl()
            memdata = proc.memory_info().vms / 1024**2
            memlist.append(memdata)
            execenv.print(i + 1, ":", memdata, "MB")
        view_curves(
            np.array(memlist),
            title="Memory leak test for DataLab application",
            ylabel="Memory (MB)",
        )


if __name__ == "__main__":
    memory_leak_test()
