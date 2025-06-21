# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction unit test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
import pytest
from guidata.qthelpers import exec_dialog, qt_app_context

import sigima_.computation.image as sigima_image
import sigima_.obj
import sigima_.param
from cdl.env import execenv
from cdl.gui.profiledialog import ProfileExtractionDialog
from cdl.tests.data import create_noisygauss_image, create_sincos_image
from cdl.utils.tests import check_array_result


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


@pytest.mark.validation
def test_line_profile() -> None:
    """Test line profile computation"""
    width, height = 256, 128
    dtype = sigima_.obj.ImageDatatypes.UINT16
    newparam = sigima_.obj.NewImageParam.create(dtype=dtype, height=height, width=width)
    ima = create_sincos_image(newparam)

    # Test horizontal line profile
    row = 100
    param = sigima_.param.LineProfileParam.create(row=row, direction="horizontal")
    sig = sigima_image.line_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == width
    exp = np.array(ima.data[row, :], dtype=float)
    check_array_result("Horizontal line profile", sig.y, exp)

    # Test vertical line profile
    col = 50
    param = sigima_.param.LineProfileParam.create(col=col, direction="vertical")
    sig = sigima_image.line_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == height
    exp = np.array(ima.data[:, col], dtype=float)
    check_array_result("Vertical line profile", sig.y, exp)


@pytest.mark.validation
def test_segment_profile() -> None:
    """Test segment profile computation"""
    width, height = 256, 128
    dtype = sigima_.obj.ImageDatatypes.UINT16
    newparam = sigima_.obj.NewImageParam.create(dtype=dtype, height=height, width=width)
    ima = create_sincos_image(newparam)

    # Test segment profile
    row1, col1, row2, col2 = 10, 20, 200, 20
    param = sigima_.param.SegmentProfileParam.create(
        row1=row1, col1=col1, row2=row2, col2=col2
    )
    sig = sigima_image.segment_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == min(row2, height - 1) - max(row1, 0) + 1
    exp = np.array(ima.data[10:200, 20], dtype=float)
    check_array_result("Segment profile", sig.y, exp)


@pytest.mark.validation
def test_average_profile() -> None:
    """Test average profile computation"""
    width, height = 256, 128
    dtype = sigima_.obj.ImageDatatypes.UINT16
    newparam = sigima_.obj.NewImageParam.create(dtype=dtype, height=height, width=width)
    ima = create_sincos_image(newparam)
    row1, col1, row2, col2 = 10, 20, 200, 230
    param = sigima_.param.AverageProfileParam.create(
        row1=row1, col1=col1, row2=row2, col2=col2
    )

    # Test horizontal average profile
    param.direction = "horizontal"
    sig = sigima_image.average_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == col2 - col1 + 1
    exp = np.array(ima.data[row1 : row2 + 1, col1 : col2 + 1].mean(axis=0), dtype=float)
    check_array_result("Horizontal average profile", sig.y, exp)

    # Test vertical average profile
    param.direction = "vertical"
    sig = sigima_image.average_profile(ima, param)
    assert sig is not None
    assert len(sig.y) == min(row2, height - 1) - max(row1, 0) + 1
    exp = np.array(ima.data[row1 : row2 + 1, col1 : col2 + 1].mean(axis=1), dtype=float)
    check_array_result("Vertical average profile", sig.y, exp)


if __name__ == "__main__":
    test_profile_unit()
    test_line_profile()
    test_segment_profile()
    test_average_profile()
