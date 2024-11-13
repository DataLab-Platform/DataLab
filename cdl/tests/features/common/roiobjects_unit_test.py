# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI objects unit tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
from guidata.qthelpers import qt_app_context

from cdl.core.gui.roieditor import plot_item_to_single_roi
from cdl.env import execenv
from cdl.obj import (
    ImageObj,
    ImageROI,
    SignalObj,
    SignalROI,
    create_image_roi,
    create_signal_roi,
)
from cdl.tests.data import create_multigauss_image, create_paracetamol_signal

CLASS_NAME = "class_name"


def __conversion_methods(roi: SignalROI | ImageROI, obj: SignalObj | ImageObj) -> None:
    """Test conversion methods for single ROI objects"""
    execenv.print("    test `to_dict` and `from_dict` methods")
    roi_dict = roi.to_dict()
    roi_new = obj.get_roi_class().from_dict(roi_dict)
    assert roi.get_single_roi(0) == roi_new.get_single_roi(0)

    execenv.print("    test `to_plot_item` and `from_plot_item` methods: ", end="")
    single_roi = roi.get_single_roi(0)
    with qt_app_context(exec_loop=False):
        plot_item = single_roi.to_plot_item(obj)
        sroi_new = plot_item_to_single_roi(plot_item)
        orig_coords = [float(val) for val in single_roi.get_physical_coords(obj)]
        new_coords = [float(val) for val in sroi_new.get_physical_coords(obj)]
        execenv.print(f"{orig_coords} --> {new_coords}")
        assert np.array_equal(orig_coords, new_coords)


def test_create_signal_roi() -> None:
    """Test create_signal_roi"""

    # ROI coordinates: for each ROI type, the coordinates are given for indices=True
    # and indices=False (physical coordinates)
    coords = {
        "segment": {
            CLASS_NAME: "SegmentROI",
            True: [50, 100],  # indices [x0, dx]
            False: [7.5, 10.0],  # physical
        },
    }

    obj = create_paracetamol_signal()

    for indices in (True, False):
        execenv.print("indices:", indices)

        for geometry in coords:
            execenv.print("  geometry:", geometry)

            roi = create_signal_roi(coords[geometry][indices], indices=indices)

            sroi = roi.get_single_roi(0)
            assert sroi.__class__.__name__ == coords[geometry][CLASS_NAME]

            cds_ind = [int(val) for val in sroi.get_indices_coords(obj)]
            assert cds_ind == coords[geometry][True]

            cds_phys = [float(val) for val in sroi.get_physical_coords(obj)]
            assert cds_phys == coords[geometry][False]

            execenv.print("    get_physical_coords:", cds_phys)
            execenv.print("    get_indices_coords: ", cds_ind)

            __conversion_methods(roi, obj)


def test_create_image_roi() -> None:
    """Test create_image_roi"""

    # ROI coordinates: for each ROI type, the coordinates are given for indices=True
    # and indices=False (physical coordinates)
    coords = {
        "rectangle": {
            CLASS_NAME: "RectangularROI",
            True: [500, 750, 1000, 1250],  # indices [x0, y0, dx, dy]
            False: [500.5, 750.5, 1000.0, 1250.0],  # physical
        },
        "circle": {
            CLASS_NAME: "CircularROI",
            True: [1500, 1500, 500],  # indices [x0, y0, radius]
            False: [1500.5, 1500.5, 500.0],  # physical
        },
    }

    obj = create_multigauss_image()

    for indices in (True, False):
        execenv.print("indices:", indices)

        for geometry in coords:
            execenv.print("  geometry:", geometry)

            roi = create_image_roi(geometry, coords[geometry][indices], indices=indices)

            sroi = roi.get_single_roi(0)
            assert sroi.__class__.__name__ == coords[geometry][CLASS_NAME]

            bbox_phys = [float(val) for val in sroi.get_bounding_box(obj)]
            x0, y0, x1, y1 = obj.physical_to_indices(bbox_phys)
            if geometry == "rectangle":
                bbox_ind = [int(xy) for xy in [x0, y0, x1 - x0, y1 - y0]]
            else:
                bbox_ind = [
                    int(xy) for xy in [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2]
                ]
            assert bbox_ind == coords[geometry][True]

            cds_phys = [float(val) for val in sroi.get_physical_coords(obj)]
            assert cds_phys == coords[geometry][False]
            cds_ind = [int(val) for val in sroi.get_indices_coords(obj)]
            assert cds_ind == coords[geometry][True]

            execenv.print("    get_bounding_box:   ", bbox_phys)
            execenv.print("    get_physical_coords:", cds_phys)
            execenv.print("    get_indices_coords: ", cds_ind)

            __conversion_methods(roi, obj)


if __name__ == "__main__":
    test_create_signal_roi()
    test_create_image_roi()
