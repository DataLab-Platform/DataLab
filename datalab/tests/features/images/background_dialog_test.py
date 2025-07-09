# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image background dialog unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import numpy as np
import sigima.objects
import sigima.params
import sigima.proc.image as sigima_image
from guidata.qthelpers import exec_dialog, qt_app_context, qt_wait_until
from sigima.tests import vistools
from sigima.tests.data import create_noisygauss_image

from datalab.env import execenv
from datalab.widgets.imagebackground import ImageBackgroundDialog


def wait_for_rect_coords(dlg: ImageBackgroundDialog) -> bool:
    """Condition to wait for rectangle coordinates."""
    # Wait for rectangle coordinates to be set. This is necessary because
    # when executing the dialog in unattended mode, the rectangle
    # coordinates may not be set immediately (Qt events may not be processed).
    try:
        dlg.get_rect_coords()
        return True
    except ValueError:
        return False


def test_image_background_selection() -> None:
    """Image background selection test."""
    with qt_app_context():
        img = create_noisygauss_image()
        dlg = ImageBackgroundDialog(img)
        dlg.resize(640, 480)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        exec_dialog(dlg)
        qt_wait_until(lambda: wait_for_rect_coords(dlg), timeout=5.0, interval=0.1)
        execenv.print(f"background: {dlg.get_background()}")
        execenv.print(f"rect coords: {dlg.get_rect_coords()}")
        # Check background value:
        x0, y0, x1, y1 = dlg.get_rect_coords()
        ix0, iy0, ix1, iy1 = dlg.imageitem.get_closest_index_rect(x0, y0, x1, y1)
        assert np.isclose(img.data[iy0:iy1, ix0:ix1].mean(), dlg.get_background())


def test_image_offset_correction_with_background_dialog() -> None:
    """Image offset correction interactive test using the background dialog."""
    with qt_app_context():
        i1 = create_noisygauss_image()
        dlg = ImageBackgroundDialog(i1)
        ok = exec_dialog(dlg)
        if ok:
            qt_wait_until(lambda: wait_for_rect_coords(dlg), timeout=5.0, interval=0.1)
            param = sigima.objects.ROI2DParam()
            # pylint: disable=unbalanced-tuple-unpacking
            ix0, iy0, ix1, iy1 = i1.physical_to_indices(dlg.get_rect_coords())
            param.x0, param.y0, param.dx, param.dy = ix0, iy0, ix1 - ix0, iy1 - iy0
            i2 = sigima_image.offset_correction(i1, param)
            i3 = sigima_image.clip(i2, sigima.params.ClipParam.create(lower=0))
            vistools.view_images_side_by_side(
                [i1, i3],
                titles=["Original image", "Corrected image"],
                title="Image offset correction and thresholding",
            )


if __name__ == "__main__":
    test_image_background_selection()
    test_image_offset_correction_with_background_dialog()
