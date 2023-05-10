# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Remote client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import os.path as osp
import time

import numpy as np

from cdl import app
from cdl.core.gui.processor.signal import XYCalibrateParam
from cdl.env import execenv
from cdl.remotecontrol import RemoteClient
from cdl.tests.data import create_2d_gaussian, create_test_signal1
from cdl.utils.tests import exec_script, temporary_directory

SHOW = False  # Show test in GUI-based test launcher


def multiple_commands(remote: RemoteClient):
    """Execute multiple XML-RPC commands"""
    with temporary_directory() as tmpdir:
        x, y = create_test_signal1().get_data()
        remote.add_signal("tutu", x, y)

        z = create_2d_gaussian(2000, np.uint16)
        remote.add_image("toto", z)

        fname = osp.join(tmpdir, osp.basename("remote_test.h5"))
        remote.save_to_h5_file(fname)
        remote.reset_all()
        remote.open_h5_files([fname], True, False)
        remote.import_h5_file(fname, True)
        remote.switch_to_signal_panel()
        remote.calc("log10")

        param = XYCalibrateParam()
        param.a, param.b = 1.2, 0.1
        remote.calc("compute_calibration", param)

        time.sleep(2)  # Avoid permission error when trying to clean-up temporary files


def test():
    """Remote client test"""
    execenv.print("Launching DataLab in a separate process")
    exec_script(app.__file__, wait=False)
    remote = RemoteClient()
    remote.try_and_connect()
    execenv.print("Executing multiple commands...", end="")
    multiple_commands(remote)
    execenv.print("OK")


if __name__ == "__main__":
    test()
