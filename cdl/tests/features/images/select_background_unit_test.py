# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image background selection unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
from guidata.qthelpers import exec_dialog, qt_app_context

from cdl.env import execenv
from cdl.widgets.imagebackground import ImageBackgroundDialog
from sigima_.tests.data import create_noisygauss_image


def test_image_background_selection():
    """Image background selection test."""
    with qt_app_context():
        img = create_noisygauss_image()
        dlg = ImageBackgroundDialog(img)
        dlg.resize(640, 480)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        with execenv.context(delay=200):
            # For more details about the why of the delay, see the comment in
            # cdl\tests\features\image\offset_correction_unit_test.py
            exec_dialog(dlg)
        execenv.print(f"background: {dlg.get_background()}")
        execenv.print(f"rect coords: {dlg.get_rect_coords()}")
        # Check background value:
        x0, y0, x1, y1 = dlg.get_rect_coords()
        ix0, iy0, ix1, iy1 = dlg.imageitem.get_closest_index_rect(x0, y0, x1, y1)
        assert np.isclose(img.data[iy0:iy1, ix0:ix1].mean(), dlg.get_background())


if __name__ == "__main__":
    test_image_background_selection()
