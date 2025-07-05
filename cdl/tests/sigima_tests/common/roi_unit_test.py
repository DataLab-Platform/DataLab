# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI creation and conversion unit tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from typing import Generator

import sigima_.obj
from sigima_.computation import image as sigima_image
from sigima_.computation import signal as sigima_signal
from sigima_.tests.data import create_multigauss_image, create_paracetamol_signal
from sigima_.tests.env import execenv

CLASS_NAME = "class_name"


def create_test_signal_rois(
    obj: sigima_.obj.SignalObj,
) -> Generator[sigima_.obj.SignalROI, None, None]:
    """Create test signal ROIs (sigima_.obj.SignalROI test object)

    Yields:
        SignalROI object
    """
    # ROI coordinates: for each ROI type, the coordinates are given for indices=True
    # and indices=False (physical coordinates)
    roi_coords = {
        "segment": {
            CLASS_NAME: "SegmentROI",
            True: [50, 100],  # indices [x0, dx]
            False: [7.5, 10.0],  # physical
        },
    }
    for indices in (True, False):
        execenv.print("indices:", indices)

        for geometry, coords in roi_coords.items():
            execenv.print("  geometry:", geometry)

            roi = sigima_.obj.create_signal_roi(coords[indices], indices=indices)

            sroi = roi.get_single_roi(0)
            assert sroi.__class__.__name__ == coords[CLASS_NAME]

            cds_ind = [int(val) for val in sroi.get_indices_coords(obj)]
            assert cds_ind == coords[True]

            cds_phys = [float(val) for val in sroi.get_physical_coords(obj)]
            assert cds_phys == coords[False]

            execenv.print("    get_physical_coords:", cds_phys)
            execenv.print("    get_indices_coords: ", cds_ind)

            yield roi


def create_test_image_rois(
    obj: sigima_.obj.ImageObj,
) -> Generator[sigima_.obj.ImageROI, None, None]:
    """Create test image ROIs (sigima_.obj.ImageROI test object)

    Yields:
        ImageROI object
    """
    # ROI coordinates: for each ROI type, the coordinates are given for indices=True
    # and indices=False (physical coordinates)
    roi_coords = {
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
        "polygon": {
            CLASS_NAME: "PolygonalROI",
            True: [450, 150, 1300, 350, 1250, 950, 400, 1350],  # indices [x0, y0, ,...]
            False: [450.5, 150.5, 1300.5, 350.5, 1250.5, 950.5, 400.5, 1350.5],  # phys.
        },
    }
    for indices in (True, False):
        execenv.print("indices:", indices)

        for geometry, coords in roi_coords.items():
            execenv.print("  geometry:", geometry)

            roi = sigima_.obj.create_image_roi(
                geometry, coords[indices], indices=indices
            )

            sroi = roi.get_single_roi(0)
            assert sroi.__class__.__name__ == coords[CLASS_NAME]

            bbox_phys = [float(val) for val in sroi.get_bounding_box(obj)]
            if geometry in ("rectangle", "circle"):
                # pylint: disable=unbalanced-tuple-unpacking
                x0, y0, x1, y1 = obj.physical_to_indices(bbox_phys)
                if geometry == "rectangle":
                    coords_from_bbox = [int(xy) for xy in [x0, y0, x1 - x0, y1 - y0]]
                else:
                    coords_from_bbox = [
                        int(xy) for xy in [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2]
                    ]
                assert coords_from_bbox == coords[True]

            cds_phys = [float(val) for val in sroi.get_physical_coords(obj)]
            assert cds_phys == coords[False]
            cds_ind = [int(val) for val in sroi.get_indices_coords(obj)]
            assert cds_ind == coords[True]

            execenv.print("    get_bounding_box:   ", bbox_phys)
            execenv.print("    get_physical_coords:", cds_phys)
            execenv.print("    get_indices_coords: ", cds_ind)

            yield roi


def __conversion_methods(
    roi: sigima_.obj.SignalROI | sigima_.obj.ImageROI,
    obj: sigima_.obj.SignalObj | sigima_.obj.ImageObj,
) -> None:
    """Test conversion methods for single ROI objects"""
    execenv.print("    test `to_dict` and `from_dict` methods")
    roi_dict = roi.to_dict()
    roi_new = obj.get_roi_class().from_dict(roi_dict)
    assert roi.get_single_roi(0) == roi_new.get_single_roi(0)


def test_signal_roi_creation() -> None:
    """Test signal ROI creation and conversion methods"""

    # ROI coordinates: for each ROI type, the coordinates are given for indices=True
    # and indices=False (physical coordinates)
    roi_coords = {
        "segment": {
            CLASS_NAME: "SegmentROI",
            True: [50, 100],  # indices [x0, dx]
            False: [7.5, 10.0],  # physical
        },
    }

    obj = create_paracetamol_signal()

    for indices in (True, False):
        execenv.print("indices:", indices)

        for geometry, coords in roi_coords.items():
            execenv.print("  geometry:", geometry)

            roi = sigima_.obj.create_signal_roi(coords[indices], indices=indices)

            sroi = roi.get_single_roi(0)
            assert sroi.__class__.__name__ == coords[CLASS_NAME]

            cds_ind = [int(val) for val in sroi.get_indices_coords(obj)]
            assert cds_ind == coords[True]

            cds_phys = [float(val) for val in sroi.get_physical_coords(obj)]
            assert cds_phys == coords[False]

            execenv.print("    get_physical_coords:", cds_phys)
            execenv.print("    get_indices_coords: ", cds_ind)

            __conversion_methods(roi, obj)


def test_signal_roi_merge() -> None:
    """Test signal ROI merge"""

    # Create a signal object with a single ROI, and another one with another ROI.
    # Compute the average of the two objects, and check if the resulting object
    # has the expected ROI (i.e. the union of the original object's ROI).

    obj1 = create_paracetamol_signal()
    obj2 = create_paracetamol_signal()
    obj2.roi = sigima_.obj.create_signal_roi([60, 120], indices=True)
    obj1.roi = sigima_.obj.create_signal_roi([50, 100], indices=True)

    # Compute the average of the two objects
    obj3 = sigima_signal.average([obj1, obj2])
    assert obj3.roi is not None, "Merged object should have a ROI"
    assert len(obj3.roi) == 2, "Merged object should have two single ROIs"
    for single_roi in obj3.roi:
        assert single_roi.get_indices_coords(obj3) in ([50, 100], [60, 120]), (
            "Merged object should have the union of the original object's ROIs"
        )


def test_image_roi_creation() -> None:
    """Test image ROI creation and conversion methods"""

    # ROI coordinates: for each ROI type, the coordinates are given for indices=True
    # and indices=False (physical coordinates)
    roi_coords = {
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
        "polygon": {
            CLASS_NAME: "PolygonalROI",
            True: [450, 150, 1300, 350, 1250, 950, 400, 1350],  # indices [x0, y0, ,...]
            False: [450.5, 150.5, 1300.5, 350.5, 1250.5, 950.5, 400.5, 1350.5],  # phys.
        },
    }

    obj = create_multigauss_image()

    for indices in (True, False):
        execenv.print("indices:", indices)

        for geometry, coords in roi_coords.items():
            execenv.print("  geometry:", geometry)

            roi = sigima_.obj.create_image_roi(
                geometry, coords[indices], indices=indices
            )

            sroi = roi.get_single_roi(0)
            assert sroi.__class__.__name__ == coords[CLASS_NAME]

            bbox_phys = [float(val) for val in sroi.get_bounding_box(obj)]
            if geometry in ("rectangle", "circle"):
                # pylint: disable=unbalanced-tuple-unpacking
                x0, y0, x1, y1 = obj.physical_to_indices(bbox_phys)
                if geometry == "rectangle":
                    coords_from_bbox = [int(xy) for xy in [x0, y0, x1 - x0, y1 - y0]]
                else:
                    coords_from_bbox = [
                        int(xy) for xy in [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2]
                    ]
                assert coords_from_bbox == coords[True]

            cds_phys = [float(val) for val in sroi.get_physical_coords(obj)]
            assert cds_phys == coords[False]
            cds_ind = [int(val) for val in sroi.get_indices_coords(obj)]
            assert cds_ind == coords[True]

            execenv.print("    get_bounding_box:   ", bbox_phys)
            execenv.print("    get_physical_coords:", cds_phys)
            execenv.print("    get_indices_coords: ", cds_ind)

            __conversion_methods(roi, obj)


def test_image_roi_merge() -> None:
    """Test image ROI merge"""

    # Create an image object with a single ROI, and another one with another ROI.
    # Compute the average of the two objects, and check if the resulting object
    # has the expected ROI (i.e. the union of the original object's ROI).

    obj1 = create_multigauss_image()
    obj2 = create_multigauss_image()
    obj2.roi = sigima_.obj.create_image_roi("rectangle", [600, 800, 1000, 1200])
    obj1.roi = sigima_.obj.create_image_roi("rectangle", [500, 750, 1000, 1250])

    # Compute the average of the two objects
    obj3 = sigima_image.average([obj1, obj2])
    assert obj3.roi is not None, "Merged object should have a ROI"
    assert len(obj3.roi) == 2, "Merged object should have two single ROIs"
    for single_roi in obj3.roi:
        assert single_roi.get_indices_coords(obj3) in (
            [500, 750, 1000, 1250],
            [600, 800, 1000, 1200],
        ), "Merged object should have the union of the original object's ROIs"


if __name__ == "__main__":
    test_signal_roi_creation()
    test_signal_roi_merge()
    test_image_roi_merge()
    test_image_roi_creation()
