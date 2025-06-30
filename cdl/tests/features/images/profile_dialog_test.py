# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction unit test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from guidata.qthelpers import exec_dialog, qt_app_context

import sigima_.param
from cdl.env import execenv
from cdl.gui.profiledialog import ProfileExtractionDialog
from sigima_.tests.data import create_noisygauss_image


def test_profile_unit():
    """Run profile extraction test"""
    with qt_app_context():
        obj = create_noisygauss_image(center=(0.0, 0.0), add_annotations=False)
        for mode in ("line", "segment", "rectangle"):
            for initial_param in (True, False):
                if initial_param:
                    if mode == "line":
                        param = sigima_.param.LineProfileParam.create(row=100, col=200)
                    elif mode == "segment":
                        param = sigima_.param.SegmentProfileParam.create(
                            row1=10, col1=20, row2=200, col2=300
                        )
                    else:
                        param = sigima_.param.AverageProfileParam.create(
                            row1=10, col1=20, row2=200, col2=300
                        )
                else:
                    if mode == "line":
                        param = sigima_.param.LineProfileParam()
                    elif mode == "segment":
                        param = sigima_.param.SegmentProfileParam()
                    else:
                        param = sigima_.param.AverageProfileParam()
                execenv.print("-" * 80)
                execenv.print(f"Testing mode: {mode} - initial_param: {initial_param}")
                dialog = ProfileExtractionDialog(
                    mode, param, add_initial_shape=initial_param
                )
                dialog.set_obj(obj)
                if initial_param:
                    dialog.edit_values()
                ok = exec_dialog(dialog)
                execenv.print(f"Returned code: {ok}")
                execenv.print(f"Param: {param}")


if __name__ == "__main__":
    test_profile_unit()
