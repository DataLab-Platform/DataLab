# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction unit test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.params
from guidata.qthelpers import exec_dialog, qt_app_context
from sigima.tests.data import create_noisy_gaussian_image

from datalab.env import execenv
from datalab.gui.profiledialog import ProfileExtractionDialog


def test_profile_unit():
    """Run profile extraction test"""
    with qt_app_context():
        obj = create_noisy_gaussian_image(center=(0.0, 0.0), add_annotations=False)
        for mode in ("line", "segment", "rectangle"):
            for initial_param in (True, False):
                if initial_param:
                    if mode == "line":
                        param = sigima.params.LineProfileParam.create(row=100, col=200)
                    elif mode == "segment":
                        param = sigima.params.SegmentProfileParam.create(
                            row1=10, col1=20, row2=200, col2=300
                        )
                    else:
                        param = sigima.params.AverageProfileParam.create(
                            row1=10, col1=20, row2=200, col2=300
                        )
                else:
                    if mode == "line":
                        param = sigima.params.LineProfileParam()
                    elif mode == "segment":
                        param = sigima.params.SegmentProfileParam()
                    else:
                        param = sigima.params.AverageProfileParam()
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
