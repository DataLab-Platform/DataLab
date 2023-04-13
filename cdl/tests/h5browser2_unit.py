# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
HDF5 browser test 2

Testing for memory leak
"""

import os

import numpy as np
import psutil

from cdl.env import execenv
from cdl.tests.data import get_test_fnames
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.vistools import view_curves
from cdl.widgets.h5browser import H5BrowserDialog

SHOW = True  # Show test in GUI-based test launcher


def memoryleak_test(fname, iterations=20):
    """Memory leak test"""
    with qt_app_context():
        proc = psutil.Process(os.getpid())
        fname = get_test_fnames(fname)[0]
        dlg = H5BrowserDialog(None)
        memlist = []
        for i in range(iterations):
            dlg.setup(fname)
            memdata = proc.memory_info().vms / 1024**2
            memlist.append(memdata)
            execenv.print(i + 1, ":", memdata, "MB")
            dlg.browser.tree.select_all(True)
            dlg.browser.tree.toggle_all(True)
            execenv.print(i + 1, ":", proc.memory_info().vms / 1024**2, "MB")
            dlg.show()
            dlg.accept()
            dlg.close()
            execenv.print(i + 1, ":", proc.memory_info().vms / 1024**2, "MB")
            dlg.cleanup()
        view_curves(
            np.array(memlist),
            title="Memory leak test for HDF5 browser dialog",
            ylabel="Memory (MB)",
        )


if __name__ == "__main__":
    memoryleak_test("scenario*.h5")
