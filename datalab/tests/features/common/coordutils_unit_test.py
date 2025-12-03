# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Coordinate utilities unit tests
"""

import numpy as np
import pytest
from sigima.objects import (
    CircularROI,
    PolygonalROI,
    RectangularROI,
    SegmentROI,
    create_image,
    create_signal,
)

from datalab.adapters_plotpy.coordutils import (
    round_image_coords,
    round_image_roi_param,
    round_signal_coords,
    round_signal_roi_param,
)


def test_round_signal_coords():
    """Test signal coordinate rounding"""
    # Create a signal with sampling period of 0.1
    x = np.arange(0, 10, 0.1)
    y = np.sin(x)
    sig = create_signal("test", x, y)

    # Test basic rounding
    coords = [1.23456789, 5.87654321]
    rounded = round_signal_coords(sig, coords)
    # With sampling period 0.1 and precision_factor 0.1, precision = 0.01
    # Should round to 2 decimal places
    assert rounded == [1.23, 5.88]

    # Test with custom precision factor
    rounded = round_signal_coords(sig, coords, precision_factor=1.0)
    # precision = 0.1, should round to 1 decimal place
    assert rounded == [1.2, 5.9]

    # Test with signal that has too few points
    sig_short = create_signal("test", np.array([1.0]), np.array([2.0]))
    coords = [1.23456789]
    rounded = round_signal_coords(sig_short, coords)
    # Should return coords as-is
    assert rounded == coords

    # Test with constant x (zero sampling period)
    sig_const = create_signal("test", np.ones(10), np.ones(10))
    rounded = round_signal_coords(sig_const, coords)
    # Should return coords as-is
    assert rounded == coords


def test_round_image_coords():
    """Test image coordinate rounding"""
    # Create an image with dx=dy=1.0 (uniform)
    data = np.ones((100, 100))
    img = create_image("test", data)

    # Test basic rounding
    coords = [10.123456, 20.987654, 30.555555, 40.444444]
    rounded = round_image_coords(img, coords)
    # With pixel spacing 1.0 and precision_factor 0.1, precision = 0.1
    # Should round to 1 decimal place
    assert rounded == [10.1, 21.0, 30.6, 40.4]

    # Test with custom precision factor
    rounded = round_image_coords(img, coords, precision_factor=1.0)
    # precision = 1.0, should round to 0 decimal places
    assert rounded == [10.0, 21.0, 31.0, 40.0]

    # Test with empty coords
    assert round_image_coords(img, []) == []

    # Test error for odd number of coordinates
    with pytest.raises(ValueError, match="even number of elements"):
        round_image_coords(img, [1.0, 2.0, 3.0])


def test_round_signal_roi_param():
    """Test signal ROI parameter rounding"""
    # Create a signal with sampling period of 0.1
    x = np.arange(0, 10, 0.1)
    y = np.sin(x)
    sig = create_signal("test", x, y)

    # Create a segment ROI
    roi = SegmentROI([1.23456789, 5.87654321], False)
    param = roi.to_param(sig, 0)

    # Round the parameter
    round_signal_roi_param(sig, param)

    # Check that coordinates are rounded
    assert param.xmin == 1.23
    assert param.xmax == 5.88


def test_round_image_roi_param_rectangle():
    """Test image ROI parameter rounding for rectangular ROI"""
    # Create an image with dx=dy=1.0
    data = np.ones((100, 100))
    img = create_image("test", data)

    # Create a rectangular ROI with floating-point errors
    roi = RectangularROI([10.0, 20.0, 50.29999999999995, 75.19999999999999], False)
    param = roi.to_param(img, 0)

    # Verify we have the floating-point errors before rounding
    assert param.dx == 50.29999999999995
    assert param.dy == 75.19999999999999

    # Round the parameter
    round_image_roi_param(img, param)

    # Check that coordinates are rounded
    assert param.x0 == 10.0
    assert param.y0 == 20.0
    assert param.dx == 50.3
    assert param.dy == 75.2


def test_round_image_roi_param_circle():
    """Test image ROI parameter rounding for circular ROI"""
    # Create an image with dx=dy=1.0
    data = np.ones((100, 100))
    img = create_image("test", data)

    # Create a circular ROI with floating-point errors
    roi = CircularROI([50.123456, 50.987654, 25.555555], False)
    param = roi.to_param(img, 0)

    # Round the parameter
    round_image_roi_param(img, param)

    # Check that coordinates are rounded
    assert param.xc == 50.1
    assert param.yc == 51.0
    assert param.r == 25.6


def test_round_image_roi_param_polygon():
    """Test image ROI parameter rounding for polygonal ROI"""
    # Create an image with dx=dy=1.0
    data = np.ones((100, 100))
    img = create_image("test", data)

    # Create a polygonal ROI with floating-point errors
    coords = [10.123456, 20.987654, 30.555555, 40.444444, 50.111111, 60.999999]
    roi = PolygonalROI(coords, False)
    param = roi.to_param(img, 0)

    # Round the parameter
    round_image_roi_param(img, param)

    # Check that coordinates are rounded
    expected = np.array([10.1, 21.0, 30.6, 40.4, 50.1, 61.0])
    np.testing.assert_array_equal(param.points, expected)


def test_round_coords_non_uniform_image():
    """Test coordinate rounding for non-uniform image coordinates"""
    # Create an image with non-uniform coordinates
    data = np.ones((10, 10))
    img = create_image("test", data)
    # Set non-uniform coordinates
    img.xcoords = np.array([0, 1, 3, 6, 10, 15, 21, 28, 36, 45])  # varying spacing
    img.ycoords = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])  # uniform spacing of 2

    # Test rounding - should use average spacing
    coords = [5.123456, 7.987654, 25.555555, 13.444444]
    rounded = round_image_coords(img, coords)

    # Average dx ≈ 5.0, average dy = 2.0
    # With precision_factor=0.1: precision_x=0.5, precision_y=0.2
    # Should round to 1 decimal place for both
    assert rounded == [5.1, 8.0, 25.6, 13.4]


def test_round_coords_preserves_structure():
    """Test that coordinate rounding preserves the structure of coordinates"""
    # Create an image
    data = np.ones((100, 100))
    img = create_image("test", data)

    # Test with multiple coordinate pairs
    coords = [
        10.111,
        20.222,
        30.333,
        40.444,
        50.555,
        60.666,
        70.777,
        80.888,
    ]
    rounded = round_image_coords(img, coords)

    # Should have same length
    assert len(rounded) == len(coords)

    # Each coordinate should be rounded independently
    assert all(isinstance(c, (int, float)) for c in rounded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
