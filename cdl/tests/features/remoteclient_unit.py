# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Remote client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: skip

import os.path as osp
import time

import numpy as np
from plotpy.builder import make

from cdl import app
from cdl.env import execenv
from cdl.param import XYCalibrateParam
from cdl.proxy import RemoteCDLProxy
from cdl.tests.data import create_2d_gaussian, create_paracetamol_signal
from cdl.utils.tests import exec_script, temporary_directory


def multiple_commands(remote: RemoteCDLProxy):
    """Execute multiple XML-RPC commands"""
    with temporary_directory() as tmpdir:
        x, y = create_paracetamol_signal().get_data()
        remote.add_signal("tutu", x, y)

        z = create_2d_gaussian(2000, np.uint16)
        remote.add_image("toto", z)
        rect = make.annotated_rectangle(100, 100, 200, 200, title="Test")
        remote.add_annotations_from_items([rect])
        uuid = remote.get_sel_object_uuids()[0]
        remote.add_label_with_title(f"Image uuid: {uuid}")
        remote.select_groups([0])
        remote.select_objects([uuid])
        remote.delete_metadata()

        fname = osp.join(tmpdir, osp.basename("remote_test.h5"))
        remote.save_to_h5_file(fname)
        remote.reset_all()
        remote.open_h5_files([fname], True, False)
        remote.import_h5_file(fname, True)
        remote.switch_to_panel("signal")
        remote.calc("log10")

        param = XYCalibrateParam.create(a=1.2, b=0.1)
        remote.calc("compute_calibration", param)

        time.sleep(2)  # Avoid permission error when trying to clean-up temporary files


def test():
    """Remote client test"""
    execenv.print("Launching DataLab in a separate process")
    exec_script(app.__file__, wait=False)
    remote = RemoteCDLProxy()
    execenv.print("Executing multiple commands...", end="")
    multiple_commands(remote)
    execenv.print("OK")


if __name__ == "__main__":
    test()
