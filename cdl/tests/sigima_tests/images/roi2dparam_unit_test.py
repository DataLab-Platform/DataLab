# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI image parameters unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
import pytest

from sigima_.obj import ROI2DParam
from sigima_.tests.env import execenv


def __create_roi_2d_parameters() -> gds.DataSetGroup:
    """Create a group of ROI parameters."""
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
    return gds.DataSetGroup(params, title="ROI Parameters")


def test_roi_2d_param_unit():
    """ROI parameters unit test."""
    group = __create_roi_2d_parameters()
    for param in group.datasets:
        execenv.print(param)


@pytest.mark.gui
def test_roi_2d_param_interactive():
    """ROI parameters interactive test."""
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context():
        group = __create_roi_2d_parameters()
        if group.edit():
            for param in group.datasets:
                execenv.print(param)


if __name__ == "__main__":
    test_roi_2d_param_interactive()
