# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI image parameters unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import guidata.dataset as gds
import numpy as np
from guidata.qthelpers import qt_app_context

from cdl.core.model.image import ROI2DParam
from cdl.env import execenv


def test_roi_param_interactive():
    """ROI parameters interactive test."""
    with qt_app_context():
        p_circ = ROI2DParam("Circular")
        p_circ.geometry = "circle"
        p_circ.xc, p_circ.yc, p_circ.r = 100, 200, 50
        p_rect = ROI2DParam("Rectangular")
        p_rect.geometry = "rectangle"
        p_rect.x0, p_rect.y0, p_rect.dx, p_rect.dy = 50, 150, 150, 250
        p_poly = ROI2DParam("Polygonal")
        p_poly.geometry = "polygon"
        p_poly.points = np.array([[50, 150], [150, 150], [150, 250], [50, 250]])
        params = [p_circ, p_rect, p_poly]
        group = gds.DataSetGroup(params, title="ROI Parameters")
        if group.edit():
            for param in params:
                execenv.print(param)


if __name__ == "__main__":
    test_roi_param_interactive()
