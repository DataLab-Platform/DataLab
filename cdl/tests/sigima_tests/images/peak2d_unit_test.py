# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image peak detection test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import time

import numpy as np
import pytest

import sigima_.computation.image as sigima_image
import sigima_.obj
import sigima_.param
from sigima_.algorithms.image import get_2d_peaks_coords
from sigima_.env import execenv
from sigima_.tests.data import get_peak2d_data
from sigima_.tests.helpers import check_array_result
from sigima_.tests.vistools import view_image_items


def exec_image_peak_detection_func(data: np.ndarray) -> np.ndarray:
    """Execute image peak detection function

    Args:
        data: 2D image data

    Returns:
        2D array of peak coordinates
    """
    t0 = time.time()
    coords = get_2d_peaks_coords(data)
    dt = time.time() - t0
    execenv.print(f"Calculation time: {int(dt * 1e3):d} ms")
    execenv.print(f"  => {coords.tolist()}")
    return coords


def view_image_peak_detection(data: np.ndarray, coords: np.ndarray) -> None:
    """View image peak detection results

    Args:
        data: 2D image data
        coords: Coordinates of detected peaks (shape: (n, 2))
    """
    from plotpy.builder import make  # pylint: disable=import-outside-toplevel

    execenv.print("Peak detection results:")
    items = [make.image(data, interpolation="linear", colormap="hsv")]
    for x, y in coords:
        items.append(make.marker((x, y)))
    view_image_items(
        items, name=view_image_peak_detection.__name__, title="Peak Detection"
    )


def test_peak2d_unit():
    """2D peak detection unit test"""
    data, coords_expected = get_peak2d_data(seed=1, multi=False)
    coords = exec_image_peak_detection_func(data)
    assert coords.shape == coords_expected.shape, (
        f"Expected {coords_expected.shape[0]} peaks, got {coords.shape[0]}"
    )
    # Absolute tolerance is set to 2 pixels, as coordinates are in pixel units
    # and the algorithm may detect peaks at slightly different pixel locations
    check_array_result("Peak coords (alg.)", coords, coords_expected, atol=2, sort=True)


@pytest.mark.validation
def test_image_peak_detection():
    """2D peak detection unit test"""
    data, coords_expected = get_peak2d_data(seed=1, multi=False)
    for create_rois in (True, False):
        obj = sigima_.obj.create_image("peak2d_unit_test", data=data)
        param = sigima_.param.Peak2DDetectionParam.create(create_rois=create_rois)
        result = sigima_image.peak_detection(obj, param)
        df = result.to_dataframe()
        coords = df.to_numpy(int)
        assert coords.shape == coords_expected.shape, (
            f"Expected {coords_expected.shape[0]} peaks, got {coords.shape[0]}"
        )
        # Absolute tolerance is set to 2 pixels, as coordinates are in pixel units
        # and the algorithm may detect peaks at slightly different pixel locations
        check_array_result(
            "Peak coords (comp.)", coords, coords_expected, atol=2, sort=True
        )
        if create_rois:
            assert result.roi is not None, "ROI should be created"
            assert len(result.roi) == coords.shape[0], (
                f"Expected {coords.shape[0]} ROIs, got {len(result.roi)}"
            )
            for i, roi in enumerate(result.roi):
                # Check that ROIs are rectangles
                assert isinstance(roi, sigima_.obj.RectangularROI), (
                    f"Expected RectangularROI, got {type(roi)}"
                )
                # Check that ROIs are correctly positioned
                x0, y0, x1, y1 = roi.get_bounding_box(obj)
                x, y = coords[i]
                assert x0 <= x < x1, f"ROI {i} x0={x0}, x={x}, x1={x1} does not match"
                assert y0 <= y < y1, f"ROI {i} y0={y0}, y={y}, y1={y1} does not match"
        else:
            assert result.roi is None, "ROI should not be created"


@pytest.mark.gui
def test_peak2d_interactive():
    """2D peak detection interactive test"""
    data, _coords = get_peak2d_data(multi=False)
    coords = exec_image_peak_detection_func(data)

    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context():
        view_image_peak_detection(data, coords)


if __name__ == "__main__":
    # test_peak2d_unit()
    test_image_peak_detection()
    # test_peak2d_interactive()
