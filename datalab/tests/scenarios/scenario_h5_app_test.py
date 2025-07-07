# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

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

from sigima.params import ClipParam
from sigima.tests.data import create_noisygauss_image, create_paracetamol_signal
from sigima.tests.helpers import WorkdirRestoringTempDir

from datalab.env import execenv
from datalab.tests import cdltest_app_context
from datalab.tests.scenarios import scenario_mac_app_test
from datalab.utils.strings import save_html_diff


def test_scenario_h5():
    """Example of high-level test scenario with HDF5 file"""
    with WorkdirRestoringTempDir() as tmpdir:
        with cdltest_app_context(console=False) as win:
            # === Creating two test signals
            panel = win.signalpanel
            sig1 = create_paracetamol_signal()
            panel.add_object(sig1)
            panel.processor.run_feature("derivative")
            # === Creating two test images
            panel = win.imagepanel
            ima1 = create_noisygauss_image(add_annotations=True)
            panel.add_object(ima1)
            param = ClipParam.create(upper=ima1.data.mean())
            panel.processor.run_feature("clip", param)
            # === Creating a macro
            scode = scenario_mac_app_test.add_macro_sample(win, 0).get_code()
            scenario_mac_app_test.add_macro_sample(win, 1)
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
            lcode = win.macropanel.get_macro(1).get_code()
            if lcode != scode:
                save_html_diff(scode, lcode, "Saved code", "Loaded code", "macro0.html")
                raise AssertionError("Macro code is not the same as saved")


if __name__ == "__main__":
    test_scenario_h5()
