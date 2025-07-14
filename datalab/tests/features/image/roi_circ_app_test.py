# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Circular ROI test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from typing import Literal

import numpy as np
import sigima.objects
from skimage import draw

from datalab.env import execenv
from datalab.tests import cdltest_app_context
from datalab.tests.features.common.roi_app_test import print_obj_shapes


def create_test_image_with_roi(
    geometry: Literal["rectangle", "circle", "polygon"],
) -> sigima.objects.ImageObj:
    """Create test image with ROIs"""
    data = np.zeros((500, 750), dtype=np.uint16)
    xc, yc, r = 500, 200, 100
    rr, cc = draw.disk((yc, xc), r)
    data[rr, cc] = 10000
    data[yc + r - 20 : yc + r, xc + r - 30 : xc + r - 10] = 50000
    if geometry == "rectangle":
        coords = [xc - r, yc - r, 2 * r, 2 * r]
    elif geometry == "circle":
        coords = [xc, yc, r]
    else:
        raise NotImplementedError(f"Geometry {geometry} not implemented")
    ima = sigima.objects.create_image(f"Test image with ROI/{geometry}", data)
    ima.roi = sigima.objects.create_image_roi(
        geometry, coords, indices=True, singleobj=False
    )
    return ima


def test_roi_circ() -> None:
    """Run circular ROI application test scenario"""
    with cdltest_app_context() as win:
        execenv.print("Circular ROI test:")
        panel = win.imagepanel
        for geometry in ("rectangle", "circle"):  # model.ROI2DParam.geometries:
            ima = create_test_image_with_roi(geometry)
            panel.add_object(ima)
            print_obj_shapes(ima)
            panel.processor.run_feature("stats")
            panel.processor.run_feature("centroid")
        # Extracting ROIs:
        for obj_nb in (1, 2):
            obj = panel[obj_nb]
            panel.objview.set_current_object(obj)
            params = obj.roi.to_params(obj)
            panel.processor.run_feature("extract_roi", params=params)


if __name__ == "__main__":
    test_roi_circ()
