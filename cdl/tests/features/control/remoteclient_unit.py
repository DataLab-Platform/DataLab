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

import os
import os.path as osp
import time

import numpy as np
from guidata.qthelpers import qt_app_context
from plotpy.builder import make

from cdl import app
from cdl.env import execenv
from cdl.param import XYCalibrateParam
from cdl.proxy import RemoteProxy
from cdl.tests.data import create_2d_gaussian, create_paracetamol_signal
from cdl.utils.tests import exec_script, temporary_directory


def multiple_commands(remote: RemoteProxy):
    """Execute multiple XML-RPC commands"""
    with temporary_directory() as tmpdir:
        x, y = create_paracetamol_signal().get_data()
        remote.add_signal("tutu", x, y)

        z = create_2d_gaussian(2000, np.uint16)
        remote.add_image("toto", z)
        rect = make.annotated_rectangle(100, 100, 200, 200, title="Test")
        area = rect.get_rect()
        remote.add_annotations_from_items([rect])
        uuid = remote.get_sel_object_uuids()[0]
        items = remote.get_object_shapes()
        assert len(items) == 1 and items[0].get_rect() == area
        remote.add_label_with_title(f"Image uuid: {uuid}")
        remote.select_groups([1])
        remote.select_objects([uuid])
        remote.delete_metadata()

        fname = osp.join(tmpdir, osp.basename("remote_test.h5"))
        remote.save_to_h5_file(fname)
        remote.reset_all()
        remote.open_h5_files([fname], True, False)
        remote.import_h5_file(fname, True)
        remote.set_current_panel("signal")
        assert remote.get_current_panel() == "signal"
        remote.calc("log10")

        param = XYCalibrateParam.create(a=1.2, b=0.1)
        remote.calc("compute_calibration", param)

        time.sleep(2)  # Avoid permission error when trying to clean-up temporary files


def test_remoteclient_unit():
    """Remote client test"""
    env = os.environ.copy()
    env[execenv.DONOTQUIT_ENV] = "1"
    execenv.print("Launching DataLab in a separate process")
    exec_script(app.__file__, wait=False, env=env)
    remote = RemoteProxy()
    execenv.print("Executing multiple commands...", end="")
    with qt_app_context():  # needed for building plot items
        multiple_commands(remote)
    execenv.print("OK")


if __name__ == "__main__":
    test_remoteclient_unit()
