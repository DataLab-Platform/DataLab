# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Circular ROI test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
from skimage import draw

import cdl.param as dlp
from cdl.env import execenv
from cdl.obj import RoiDataGeometries, create_image
from cdl.tests import cdltest_app_context
from cdl.tests.features.common.roi_app_test import print_obj_shapes


def create_test_image_with_roi(roi_geometry: RoiDataGeometries):
    """Create test image with ROIs"""
    data = np.zeros((500, 750), dtype=np.uint16)
    xc, yc, r = 500, 200, 100
    rr, cc = draw.disk((yc, xc), r)
    data[rr, cc] = 10000
    data[yc + r - 20 : yc + r, xc + r - 30 : xc + r - 10] = 50000
    if roi_geometry is RoiDataGeometries.RECTANGLE:
        roi = [xc - r, yc - r, xc + r, yc + r]
        geom = "Rectangular"
    else:
        roi = [xc - r, yc, xc + r, yc]
        geom = "Circular"
    ima = create_image(f"{geom} ROI test image", data)
    ima.roi = np.array([roi], int)
    return ima


def test_roi_circ():
    """Run circular ROI application test scenario"""
    with cdltest_app_context() as win:
        execenv.print("Circular ROI test:")
        panel = win.imagepanel
        for geometry in RoiDataGeometries:
            ima = create_test_image_with_roi(geometry)
            panel.add_object(ima)
            print_obj_shapes(ima)
            panel.processor.compute_stats()
            panel.processor.compute_centroid()
        # Extracting ROIs:
        rparam = dlp.ROIDataParam.create(singleobj=False)
        for obj_nb in (1, 2):
            panel.objview.set_current_object(panel[obj_nb])
            panel.processor.compute_roi_extraction(rparam)


if __name__ == "__main__":
    test_roi_circ()
