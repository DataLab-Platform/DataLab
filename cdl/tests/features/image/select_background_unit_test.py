# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image background selection unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
from guidata.qthelpers import exec_dialog, qt_app_context

from cdl.env import execenv
from cdl.tests.data import create_noisygauss_image
from cdl.widgets.imagebackground import ImageBackgroundDialog


def test_image_background_selection():
    """Image background selection test."""
    with qt_app_context():
        img = create_noisygauss_image()
        dlg = ImageBackgroundDialog(img)
        dlg.resize(640, 480)
        dlg.setObjectName(dlg.objectName() + "_00")  # to avoid timestamp suffix
        exec_dialog(dlg)
    execenv.print(f"background: {dlg.get_background()}")
    execenv.print(f"index range: {dlg.get_index_range()}")
    # Check background value:
    x0, y0, x1, y1 = dlg.get_index_range()
    assert np.isclose(img.data[y0:y1, x0:x1].mean(), dlg.get_background())


if __name__ == "__main__":
    test_image_background_selection()
