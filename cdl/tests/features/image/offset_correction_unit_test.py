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
from guidata.qthelpers import exec_dialog, qt_app_context

import cdl.core.computation.image as cpi
import cdl.param
from cdl.obj import ROI2DParam
from cdl.tests.data import create_noisygauss_image
from cdl.utils.vistools import view_images_side_by_side
from cdl.widgets.imagebackground import ImageBackgroundDialog


def test_image_offset_correction_interactive() -> None:
    """Image offset correction interactive test."""
    with qt_app_context():
        i1 = create_noisygauss_image()
        dlg = ImageBackgroundDialog(i1)
        if exec_dialog(dlg):
            param = ROI2DParam()
            param.x0, param.y0, param.x1, param.y1 = dlg.get_index_range()
            i2 = cpi.compute_offset_correction(i1, param)
            i3 = cpi.compute_threshold(i2, cdl.param.ClipParam.create(value=0))
            view_images_side_by_side(
                [i1.make_item(), i3.make_item()],
                titles=["Original image", "Corrected image"],
                title="Image offset correction and thresholding",
            )


@pytest.mark.validation
def test_image_offset_correction() -> None:
    """Image offset correction validation test."""
    i1 = create_noisygauss_image()
    param = ROI2DParam.create(x0=0, y0=0, x1=10, y1=10)
    i2 = cpi.compute_offset_correction(i1, param)

    # Check that the offset correction has been applied
    x0, y0, x1, y1 = param.x0, param.y0, param.x1, param.y1
    offset = np.mean(i1.data[y0:y1, x0:x1])
    assert np.allclose(i2.data, i1.data - offset), "Offset correction failed"


if __name__ == "__main__":
    test_image_offset_correction_interactive()
    test_image_offset_correction()
