# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction unit test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
import pytest

import sigima_.computation.image as sigima_image
import sigima_.obj
import sigima_.param
from sigima_.tests.data import create_sincos_image
from sigima_.tests.helpers import check_array_result


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
    test_line_profile()
    test_segment_profile()
    test_average_profile()
