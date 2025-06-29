# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image background dialog unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

from guidata.qthelpers import exec_dialog, qt_app_context

import sigima_.computation.image as sigima_image
import sigima_.obj
import sigima_.param
from cdl.env import execenv
from cdl.widgets.imagebackground import ImageBackgroundDialog
from sigima_.tests import vistools
from sigima_.tests.data import create_noisygauss_image


def test_image_background_dialog() -> None:
    """Image background dialog test."""
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
            # drawn, the background value is not computed, and `get_rect_coords()`
            # returns `None` which causes the test to fail.
            ok = exec_dialog(dlg)
        if ok:
            param = sigima_.obj.ROI2DParam()
            # pylint: disable=unbalanced-tuple-unpacking
            ix0, iy0, ix1, iy1 = i1.physical_to_indices(dlg.get_rect_coords())
            param.x0, param.y0, param.dx, param.dy = ix0, iy0, ix1 - ix0, iy1 - iy0
            i2 = sigima_image.offset_correction(i1, param)
            i3 = sigima_image.clip(i2, sigima_.param.ClipParam.create(lower=0))
            vistools.view_images_side_by_side(
                [i1, i3],
                titles=["Original image", "Corrected image"],
                title="Image offset correction and thresholding",
            )


if __name__ == "__main__":
    test_image_background_dialog()
