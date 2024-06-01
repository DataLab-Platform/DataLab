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
from cdl.env import execenv
from cdl.obj import ROI2DParam
from cdl.tests.data import create_noisygauss_image
from cdl.utils.vistools import view_images_side_by_side
from cdl.widgets.imagebackground import ImageBackgroundDialog


def test_image_offset_correction_interactive() -> None:
    """Image offset correction interactive test."""
    with qt_app_context():
        i1 = create_noisygauss_image()
        dlg = ImageBackgroundDialog(i1)
        with execenv.context(delay=200):
            # On Windows, the `QApplication.processEvents()` introduced with
            # guidata V3.5.1 in `exec_dialog` is sufficient to force an update
            # of the dialog. The delay is not required.
            # On Linux, the delay is required to ensure that the dialog is displayed
            # because the `QApplication.processEvents()` do not trigger the drawing
            # event on the dialog as expected. So, the `RangeComputation2d` is not
            # drawn, the background value is not computed, and `get_index_range()`
            # returns `None` which causes the test to fail.
            ok = exec_dialog(dlg)
        if ok:
            param = ROI2DParam()
            param.xr0, param.yr0, param.xr1, param.yr1 = dlg.get_index_range()
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
    param = ROI2DParam.create(xr0=0, yr0=0, xr1=10, yr1=10)
    i2 = cpi.compute_offset_correction(i1, param)

    # Check that the offset correction has been applied
    x0, y0, x1, y1 = param.xr0, param.yr0, param.xr1, param.yr1
    offset = np.mean(i1.data[y0:y1, x0:x1])
    assert np.allclose(i2.data, i1.data - offset), "Offset correction failed"


if __name__ == "__main__":
    test_image_offset_correction_interactive()
    test_image_offset_correction()
