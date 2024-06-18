# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for image processing functions
-----------------------------------------

Features from the "Processing" menu are covered by this test.
The "Processing" menu contains functions to process images, such as
denoising, FFT, thresholding, etc.

Some of the functions are tested here, such as the image clipping.
Other functions may be tested in different files, depending on the
complexity of the function.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as spi
import scipy.signal as sps

import cdl.computation.image as cpi
import cdl.obj
import cdl.param
from cdl.tests.data import get_test_image
from cdl.utils.tests import check_array_result, check_scalar_result


@pytest.mark.validation
def test_image_calibration() -> None:
    """Validation test for the image calibration processing."""
    src = get_test_image("flower.npy")
    p = cdl.param.ZCalibrateParam()

    # Test with a = 1 and b = 0: should do nothing
    p.a, p.b = 1.0, 0.0
    dst = cpi.compute_calibration(src, p)
    exp = src.data
    check_array_result("Calibration[identity]", dst.data, exp)

    # Testing with random values of a and b
    p.a, p.b = 0.5, 0.1
    exp = p.a * src.data + p.b
    dst = cpi.compute_calibration(src, p)
    check_array_result(f"Calibration[a={p.a},b={p.b}]", dst.data, exp)


@pytest.mark.validation
def test_image_swap_axes() -> None:
    """Validation test for the image axes swapping processing."""
    src = get_test_image("flower.npy")
    dst = cpi.compute_swap_axes(src)
    exp = np.swapaxes(src.data, 0, 1)
    check_array_result("SwapAxes", dst.data, exp)


@pytest.mark.validation
def test_image_normalize() -> None:
    """Validation test for the image normalization processing."""
    src = get_test_image("flower.npy")
    src.data = np.array(src.data, dtype=float)
    p = cdl.param.NormalizeParam()

    # Given the fact that the normalization methods implementations are
    # straightforward, we do not need to compare arrays with each other,
    # we simply need to check if some properties are satisfied.
    for method_value, _method_name in p.methods:
        p.method = method_value
        dst = cpi.compute_normalize(src, p)
        title = f"Normalize[method='{p.method}']"
        if p.method == "maximum":
            exp_min, exp_max = src.data.min() / src.data.max(), 1.0
        elif p.method == "amplitude":
            exp_min, exp_max = 0.0, 1.0
        elif p.method == "area":
            area = src.data.sum()
            exp_min, exp_max = src.data.min() / area, src.data.max() / area
        elif p.method == "energy":
            energy = np.sqrt(np.sum(np.abs(src.data) ** 2))
            exp_min, exp_max = src.data.min() / energy, src.data.max() / energy
        elif p.method == "rms":
            rms = np.sqrt(np.mean(np.abs(src.data) ** 2))
            exp_min, exp_max = src.data.min() / rms, src.data.max() / rms
        check_scalar_result(f"{title}|min", dst.data.min(), exp_min)
        check_scalar_result(f"{title}|max", dst.data.max(), exp_max)


@pytest.mark.validation
def test_image_clip() -> None:
    """Validation test for the image clipping processing."""
    src = get_test_image("flower.npy")
    p = cdl.param.ClipParam()

    for lower, upper in ((float("-inf"), float("inf")), (50, 100)):
        p.lower, p.upper = lower, upper
        dst = cpi.compute_clip(src, p)
        exp = np.clip(src.data, p.lower, p.upper)
        check_array_result(f"Clip[{lower},{upper}]", dst.data, exp)


@pytest.mark.validation
def test_image_offset_correction() -> None:
    """Validation test for the image offset correction processing."""
    src = get_test_image("flower.npy")
    # Defining the ROI that will be used to estimate the offset
    p = cdl.obj.ROI2DParam.create(xr0=0, yr0=0, xr1=50, yr1=20)
    dst = cpi.compute_offset_correction(src, p)
    exp = src.data - np.mean(src.data[p.yr0 : p.yr1, p.xr0 : p.xr1])
    check_array_result("OffsetCorrection", dst.data, exp)


@pytest.mark.validation
def test_image_gaussian_filter() -> None:
    """Validation test for the image Gaussian filter processing."""
    src = get_test_image("flower.npy")
    for sigma in (10.0, 50.0):
        p = cdl.param.GaussianParam.create(sigma=sigma)
        dst = cpi.compute_gaussian_filter(src, p)
        exp = spi.gaussian_filter(src.data, sigma=sigma)
        check_array_result(f"GaussianFilter[sigma={sigma}]", dst.data, exp)


@pytest.mark.validation
def test_image_moving_average() -> None:
    """Validation test for the image moving average processing."""
    src = get_test_image("flower.npy")
    p = cdl.param.MovingAverageParam.create(n=30)
    for mode in p.modes:
        p.mode = mode
        dst = cpi.compute_moving_average(src, p)
        exp = spi.uniform_filter(src.data, size=p.n, mode=p.mode)
        check_array_result(f"MovingAvg[n={p.n},mode={p.mode}]", dst.data, exp)


@pytest.mark.validation
def test_image_moving_median() -> None:
    """Validation test for the image moving median processing."""
    src = get_test_image("flower.npy")
    p = cdl.param.MovingMedianParam.create(n=5)
    for mode in p.modes:
        p.mode = mode
        dst = cpi.compute_moving_median(src, p)
        exp = spi.median_filter(src.data, size=p.n, mode=p.mode)
        check_array_result(f"MovingMed[n={p.n},mode={p.mode}]", dst.data, exp)


@pytest.mark.validation
def test_image_wiener() -> None:
    """Validation test for the image Wiener filter processing."""
    src = get_test_image("flower.npy")
    dst = cpi.compute_wiener(src)
    exp = sps.wiener(src.data)
    check_array_result("Wiener", dst.data, exp)


if __name__ == "__main__":
    test_image_calibration()
    test_image_swap_axes()
    test_image_normalize()
    test_image_clip()
    test_image_offset_correction()
    test_image_gaussian_filter()
    test_image_moving_average()
    test_image_moving_median()
    test_image_wiener()
