# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab mainwindow automation test
----------------------------------

This test scenario shows how to use the DataLab application in a test
scenario. It also validates the HDF5 file format.

This scenario executes the following steps:

  - Creating two test signals
  - Creating two test images
  - Creating a macro
  - Saving current project (h5 file)
  - Removing all objects
  - Opening another project (h5 file)
  - Access current image metadata
  - Checking macro code
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import os.path as osp

from cdl.env import execenv
from cdl.param import ClipParam
from cdl.tests import cdl_app_context
from cdl.tests.data import create_noisygauss_image, create_paracetamol_signal
from cdl.tests.scenarios import scenario_mac_app
from cdl.utils.misc import save_html_diff
from cdl.utils.tests import temporary_directory


def test():
    """Example of high-level test scenario with HDF5 file"""
    with temporary_directory() as tmpdir:
        with cdl_app_context(console=False) as win:
            # === Creating two test signals
            panel = win.signalpanel
            sig1 = create_paracetamol_signal()
            panel.add_object(sig1)
            panel.processor.compute_derivative()
            # === Creating two test images
            panel = win.imagepanel
            ima1 = create_noisygauss_image()
            panel.add_object(ima1)
            param = ClipParam.create(value=ima1.data.mean())
            panel.processor.compute_clip(param)
            # === Creating a macro
            scode = scenario_mac_app.add_macro_sample(win, 0).get_code()
            scenario_mac_app.add_macro_sample(win, 1)
            # === Saving project
            fname = osp.join(tmpdir, "test.h5")
            win.save_to_h5_file(fname)
            # === Removing all objects
            for panel in win.panels:
                panel.remove_all_objects()
            # === Reopening previously saved project
            win.open_h5_files([fname], import_all=True, reset_all=True)
            # === Accessing object metadata
            obj = win.imagepanel.objmodel.get_groups()[0][0]
            execenv.print(f"Image '{obj.title}':")
            for key, value in obj.metadata.items():
                execenv.print(f'  metadata["{key}"] = {value}')
            # === Checking macro code
            lcode = win.macropanel.get_macro(0).get_code()
            if lcode != scode:
                save_html_diff(scode, lcode, "Saved code", "Loaded code", "macro0.html")
                raise AssertionError("Macro code is not the same as saved")


if __name__ == "__main__":
    test()
