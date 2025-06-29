# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image offset correction unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import numpy as np
import pytest

import sigima_.computation.image as sigima_image
import sigima_.obj
import sigima_.param
from sigima_.tests.data import create_noisygauss_image


@pytest.mark.gui
def test_image_offset_correction_interactive() -> None:
    """Image offset correction interactive test."""
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context
    from plotpy.builder import make
    from plotpy.items import RectangleShape
    from plotpy.tools import RectangleTool
    from plotpy.widgets.selectdialog import SelectDialog, select_with_shape_tool

    from sigima_.tests import vistools

    with qt_app_context():
        i1 = create_noisygauss_image()
        shape: RectangleShape = select_with_shape_tool(
            None,
            RectangleTool,
            make.image(i1.data, interpolation="nearest", eliminate_outliers=1.0),
            "Select background area",
            tooldialogclass=SelectDialog,
        )
        if shape is not None:
            param = sigima_.obj.ROI2DParam()
            # pylint: disable=unbalanced-tuple-unpacking
            ix0, iy0, ix1, iy1 = i1.physical_to_indices(shape.get_rect())
            param.x0, param.y0, param.dx, param.dy = ix0, iy0, ix1 - ix0, iy1 - iy0
            i2 = sigima_image.offset_correction(i1, param)
            i3 = sigima_image.clip(i2, sigima_.param.ClipParam.create(lower=0))
            vistools.view_images_side_by_side(
                [i1, i3],
                titles=["Original image", "Corrected image"],
                title="Image offset correction and thresholding",
            )


@pytest.mark.validation
def test_image_offset_correction() -> None:
    """Image offset correction validation test."""
    i1 = create_noisygauss_image()
    param = sigima_.obj.ROI2DParam.create(x0=0, y0=0, dx=10, dy=10)
    i2 = sigima_image.offset_correction(i1, param)

    # Check that the offset correction has been applied
    x0, y0 = param.x0, param.y0
    x1, y1 = x0 + param.dx, y0 + param.dy
    offset = np.mean(i1.data[y0:y1, x0:x1])
    assert np.allclose(i2.data, i1.data - offset), "Offset correction failed"


if __name__ == "__main__":
    test_image_offset_correction_interactive()
    test_image_offset_correction()
