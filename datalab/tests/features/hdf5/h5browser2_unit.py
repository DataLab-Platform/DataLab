# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
HDF5 browser unit tests 2
-------------------------

Testing for memory leak
"""

# guitest: show,skip

import os
import time

import numpy as np
import psutil
from guidata.qthelpers import qt_app_context
from sigima.tests.vistools import view_curves

from datalab.env import execenv
from datalab.tests import helpers
from datalab.widgets.h5browser import H5BrowserDialog


def test_memoryleak(fname, iterations=20):
    """Memory leak test"""
    with qt_app_context():
        proc = psutil.Process(os.getpid())
        fname = helpers.get_test_fnames(fname)[0]
        dlg = H5BrowserDialog(None)
        memlist = []
        for i in range(iterations):
            t0 = time.time()
            dlg.open_file(fname)
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
            execenv.print(i + 1, ":", f"{(time.time() - t0):.1f} s")
        view_curves(
            np.array(memlist),
            title="Memory leak test for HDF5 browser dialog",
            ylabel="Memory (MB)",
        )


if __name__ == "__main__":
    test_memoryleak("scenario*.h5")
