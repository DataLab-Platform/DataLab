# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for signal processing functions
------------------------------------------

Features from the "Processing" menu are covered by this test.
The "Processing" menu contains functions to process signals, such as
calibration, smoothing, and baseline correction.

Some of the functions are tested here, such as the signal calibration.
Other functions may be tested in different files, depending on the
complexity of the function.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import numpy as np
import pytest
import scipy
import scipy.ndimage as spi
import scipy.signal as sps
from packaging.version import Version

import sigima_.algorithms.coordinates as alg_coords
import sigima_.computation.signal as sigima_signal
import sigima_.obj
import sigima_.param
import sigima_.tests.data as ctd
from sigima_.tests.data import get_test_signal
from sigima_.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.validation
def test_signal_calibration() -> None:
    """Validation test for the signal calibration processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima_.param.XYCalibrateParam()

    # Test with a = 1 and b = 0: should do nothing
    p.a, p.b = 1.0, 0.0
    for axis, _taxis in p.axes:
        p.axis = axis
        dst = sigima_signal.calibration(src, p)
        exp = src.xydata
        check_array_result("Calibration[identity]", dst.xydata, exp)

    # Testing with random values of a and b
    p.a, p.b = 0.5, 0.1
    for axis, _taxis in p.axes:
        p.axis = axis
        exp_x1, exp_y1 = src.xydata.copy()
        if axis == "x":
            exp_x1 = p.a * exp_x1 + p.b
        else:
            exp_y1 = p.a * exp_y1 + p.b
        dst = sigima_signal.calibration(src, p)
        res_x1, res_y1 = dst.xydata
        title = f"Calibration[{axis},a={p.a},b={p.b}]"
        check_array_result(f"{title}.x", res_x1, exp_x1)
        check_array_result(f"{title}.y", res_y1, exp_y1)


@pytest.mark.validation
def test_signal_swap_axes() -> None:
    """Validation test for the signal axes swapping processing."""
    src = get_test_signal("paracetamol.txt")
    dst = sigima_signal.swap_axes(src)
    exp_y, exp_x = src.xydata
    check_array_result("SwapAxes|x", dst.x, exp_x)
    check_array_result("SwapAxes|y", dst.y, exp_y)


@pytest.mark.validation
def test_signal_reverse_x() -> None:
    """Validation test for the signal reverse x processing."""
    src = get_test_signal("paracetamol.txt")
    dst = sigima_signal.reverse_x(src)
    exp = src.data[::-1]
    check_array_result("ReverseX", dst.data, exp)


def test_to_polar() -> None:
    """Unit test for the cartesian to polar conversion."""
    title = "Cartesian2Polar"
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    r, theta = alg_coords.to_polar(x, y, "rad")
    exp_r = np.array([0.0, np.sqrt(2.0), np.sqrt(8.0), np.sqrt(18.0), np.sqrt(32.0)])
    exp_theta = np.array([0.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0])
    check_array_result(f"{title}|r", r, exp_r)
    check_array_result(f"{title}|theta", theta, exp_theta)

    r, theta = alg_coords.to_polar(x, y, unit="deg")
    exp_theta = np.array([0.0, 45.0, 45.0, 45.0, 45.0])
    check_array_result(f"{title}|r", r, exp_r)
    check_array_result(f"{title}|theta", theta, exp_theta)


def test_to_cartesian() -> None:
    """Unit test for the polar to cartesian conversion."""
    title = "Polar2Cartesian"
    r = np.array([0.0, np.sqrt(2.0), np.sqrt(8.0), np.sqrt(18.0), np.sqrt(32.0)])
    theta = np.array([0.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0])

    x, y = alg_coords.to_cartesian(r, theta, "rad")
    exp_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    exp_y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    check_array_result(f"{title}|x", x, exp_x)
    check_array_result(f"{title}|y", y, exp_y)

    theta = np.array([0.0, 45.0, 45.0, 45.0, 45.0])
    x, y = alg_coords.to_cartesian(r, theta, unit="deg")
    check_array_result(f"{title}|x", x, exp_x)
    check_array_result(f"{title}|y", y, exp_y)


@pytest.mark.validation
def test_signal_to_polar() -> None:
    """Validation test for the signal cartesian to polar processing."""
    title = "Cartesian2Polar"
    p = sigima_.param.AngleUnitParam()
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    src = sigima_.obj.create_signal("test", x, y)

    for p.unit, _unit_name in sigima_.param.AngleUnitParam.units:
        dst1 = sigima_signal.to_polar(src, p)
        dst2 = sigima_signal.to_cartesian(dst1, p)
        check_array_result(f"{title}|x", dst2.x, x)
        check_array_result(f"{title}|y", dst2.y, y)


@pytest.mark.validation
def test_signal_to_cartesian() -> None:
    """Validation test for the signal polar to cartesian processing."""
    title = "Polar2Cartesian"
    p = sigima_.param.AngleUnitParam()
    r = np.array([0.0, np.sqrt(2.0), np.sqrt(8.0), np.sqrt(18.0), np.sqrt(32.0)])

    angles_deg = np.array([0.0, 45.0, 45.0, 45.0, 45.0])
    angles_rad = np.array([0.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0])
    for p.unit, _unit_name in sigima_.param.AngleUnitParam.units:
        theta = angles_rad if p.unit == "rad" else angles_deg
        src = sigima_.obj.create_signal("test", r, theta)
        dst1 = sigima_signal.to_cartesian(src, p)
        dst2 = sigima_signal.to_polar(dst1, p)
        check_array_result(f"{title}|x", dst2.x, r)
        check_array_result(f"{title}|y", dst2.y, theta)


@pytest.mark.validation
def test_signal_normalize() -> None:
    """Validation test for the signal normalization processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima_.param.NormalizeParam()
    src.y[10:15] = np.nan  # Adding some NaN values to the signal

    # Given the fact that the normalization methods implementations are
    # straightforward, we do not need to compare arrays with each other,
    # we simply need to check if some properties are satisfied.
    for method_value, _method_name in p.methods:
        p.method = method_value
        dst = sigima_signal.normalize(src, p)
        title = f"Normalize[method='{p.method}']"
        exp_min, exp_max = None, None
        if p.method == "maximum":
            exp_min, exp_max = np.nanmin(src.data) / np.nanmax(src.data), 1.0
        elif p.method == "amplitude":
            exp_min, exp_max = 0.0, 1.0
        elif p.method == "area":
            area = np.nansum(src.data)
            exp_min, exp_max = np.nanmin(src.data) / area, np.nanmax(src.data) / area
        elif p.method == "energy":
            energy = np.sqrt(np.nansum(np.abs(src.data) ** 2))
            exp_min, exp_max = (
                np.nanmin(src.data) / energy,
                np.nanmax(src.data) / energy,
            )
        elif p.method == "rms":
            rms = np.sqrt(np.nanmean(np.abs(src.data) ** 2))
            exp_min, exp_max = np.nanmin(src.data) / rms, np.nanmax(src.data) / rms
        check_scalar_result(f"{title}|min", np.nanmin(dst.data), exp_min)
        check_scalar_result(f"{title}|max", np.nanmax(dst.data), exp_max)


@pytest.mark.validation
def test_signal_clip() -> None:
    """Validation test for the signal clipping processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima_.param.ClipParam()

    for lower, upper in ((float("-inf"), float("inf")), (250.0, 500.0)):
        p.lower, p.upper = lower, upper
        dst = sigima_signal.clip(src, p)
        exp = np.clip(src.data, p.lower, p.upper)
        check_array_result(f"Clip[{lower},{upper}]", dst.data, exp)


@pytest.mark.validation
def test_signal_convolution() -> None:
    """Validation test for the signal convolution processing."""
    src1 = get_test_signal("paracetamol.txt")
    snew = sigima_.obj.NewSignalParam.create(
        title="Gaussian", stype=sigima_.obj.SignalTypes.GAUSS
    )
    extra_param = sigima_.obj.GaussLorentzVoigtParam.create(sigma=10.0)
    src2 = sigima_.obj.create_signal_from_param(snew, extra_param=extra_param)

    dst = sigima_signal.convolution(src1, src2)
    exp = np.convolve(src1.data, src2.data, mode="same")
    check_array_result("Convolution", dst.data, exp)


@pytest.mark.validation
def test_signal_derivative() -> None:
    """Validation test for the signal derivative processing."""
    src = get_test_signal("paracetamol.txt")
    dst = sigima_signal.derivative(src)
    x, y = src.xydata

    # Compute the derivative using a simple finite difference:
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dydx = dy / dx
    exp = np.zeros_like(y)
    exp[0] = dydx[0]
    exp[1:-1] = (dydx[:-1] * dx[1:] + dydx[1:] * dx[:-1]) / (dx[1:] + dx[:-1])
    exp[-1] = dydx[-1]

    check_array_result("Derivative", dst.y, exp)


@pytest.mark.validation
def test_signal_integral() -> None:
    """Validation test for the signal integral processing."""
    src = get_test_signal("paracetamol.txt")
    src.data /= np.max(src.data)

    # Check the integral of the derivative:
    dst = sigima_signal.integral(sigima_signal.derivative(src))
    # The integral of the derivative should be the original signal, up to a constant:
    dst.y += src.y[0]

    check_array_result("Integral[Derivative]", dst.y, src.y, atol=0.05)

    dst = sigima_signal.integral(src)
    x, y = src.xydata

    # Compute the integral using a simple trapezoidal rule:
    exp = np.zeros_like(y)
    exp[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    exp[0] = exp[1]

    check_array_result("Integral", dst.y, exp, atol=0.05)


@pytest.mark.validation
def test_signal_detrending() -> None:
    """Validation test for the signal detrending processing."""
    src = get_test_signal("paracetamol.txt")
    for method_value, _method_name in sigima_.param.DetrendingParam.methods:
        p = sigima_.param.DetrendingParam.create(method=method_value)
        dst = sigima_signal.detrending(src, p)
        exp = sps.detrend(src.data, type=p.method)
        check_array_result(f"Detrending [method={p.method}]", dst.data, exp)


@pytest.mark.validation
def test_signal_offset_correction() -> None:
    """Validation test for the signal offset correction processing."""
    src = get_test_signal("paracetamol.txt")
    # Defining the ROI that will be used to estimate the offset
    imin, imax = 0, 20
    p = sigima_.obj.ROI1DParam.create(xmin=src.x[imin], xmax=src.x[imax])
    dst = sigima_signal.offset_correction(src, p)
    exp = src.data - np.mean(src.data[imin:imax])
    check_array_result("OffsetCorrection", dst.data, exp)


@pytest.mark.validation
def test_signal_gaussian_filter() -> None:
    """Validation test for the signal Gaussian filter processing."""
    src = get_test_signal("paracetamol.txt")
    for sigma in (10.0, 50.0):
        p = sigima_.param.GaussianParam.create(sigma=sigma)
        dst = sigima_signal.gaussian_filter(src, p)
        exp = spi.gaussian_filter(src.data, sigma=sigma)
        check_array_result(f"GaussianFilter[sigma={sigma}]", dst.data, exp)


@pytest.mark.validation
def test_signal_moving_average() -> None:
    """Validation test for the signal moving average processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima_.param.MovingAverageParam.create(n=30)
    for mode in p.modes:
        p.mode = mode
        dst = sigima_signal.moving_average(src, p)
        exp = spi.uniform_filter(src.data, size=p.n, mode=p.mode)

        # Implementation note:
        # --------------------
        #
        # The SciPy's `uniform_filter` handles the edges more accurately than
        # a method based on a simple convolution with a kernel of ones like this:
        # (the following function was the original implementation of the moving average
        # in DataLab before it was replaced by the SciPy's `uniform_filter` function)
        #
        # def moving_average(y: np.ndarray, n: int) -> np.ndarray:
        #     y_padded = np.pad(y, (n // 2, n - 1 - n // 2), mode="edge")
        #     return np.convolve(y_padded, np.ones((n,)) / n, mode="valid")

        check_array_result(f"MovingAvg[n={p.n},mode={p.mode}]", dst.data, exp, rtol=0.1)


@pytest.mark.validation
@pytest.mark.skipif(
    Version("1.15.0") <= Version(scipy.__version__) <= Version("1.15.2"),
    reason="Skipping test: scipy median_filter is broken in 1.15.0-1.15.2",
)
def test_signal_moving_median() -> None:
    """Validation test for the signal moving median processing."""
    src = get_test_signal("paracetamol.txt")
    p = sigima_.param.MovingMedianParam.create(n=15)
    for mode in p.modes:
        p.mode = mode
        dst = sigima_signal.moving_median(src, p)
        exp = spi.median_filter(src.data, size=p.n, mode=p.mode)
        check_array_result(f"MovingMed[n={p.n},mode={p.mode}]", dst.data, exp, rtol=0.1)


@pytest.mark.validation
def test_signal_wiener() -> None:
    """Validation test for the signal Wiener filter processing."""
    src = get_test_signal("paracetamol.txt")
    dst = sigima_signal.wiener(src)
    exp = sps.wiener(src.data)
    check_array_result("Wiener", dst.data, exp)


@pytest.mark.validation
def test_signal_resampling() -> None:
    """Validation test for the signal resampling processing."""
    src1 = ctd.create_periodic_signal(sigima_.obj.SignalTypes.SINUS, freq=50.0, size=5)
    x1, y1 = src1.xydata
    p1 = sigima_.param.ResamplingParam.create(
        xmin=src1.x[0], xmax=src1.x[-1], nbpts=src1.x.size
    )
    dst1 = sigima_signal.resampling(src1, p1)
    dst1x, dst1y = dst1.xydata
    check_array_result("x1new", dst1x, x1)
    check_array_result("y1new", dst1y, y1)

    src2 = ctd.create_periodic_signal(sigima_.obj.SignalTypes.SINUS, freq=50.0, size=9)
    p2 = sigima_.param.ResamplingParam.create(
        xmin=src1.x[0], xmax=src1.x[-1], nbpts=src1.x.size
    )
    dst2 = sigima_signal.resampling(src2, p2)
    dst2x, dst2y = dst2.xydata
    check_array_result("x2new", dst2x, x1)
    check_array_result("y2new", dst2y, y1)


@pytest.mark.validation
def test_signal_XY_mode() -> None:
    """Validation test for the signal X-Y mode processing."""
    s1 = ctd.create_periodic_signal(sigima_.obj.SignalTypes.COSINUS, freq=50.0, size=5)
    s2 = ctd.create_periodic_signal(sigima_.obj.SignalTypes.SINUS, freq=50.0, size=5)
    dst = sigima_signal.xy_mode(s1, s2)
    x, y = dst.xydata
    check_array_result("XYMode", x, s1.y)
    check_array_result("XYMode", y, s2.y)
    check_array_result("XYMode", x**2 + y**2, np.ones_like(x))

    s1 = ctd.create_periodic_signal(sigima_.obj.SignalTypes.COSINUS, freq=50.0, size=9)
    s2 = ctd.create_periodic_signal(sigima_.obj.SignalTypes.SINUS, freq=50.0, size=5)
    dst = sigima_signal.xy_mode(s1, s2)
    x, y = dst.xydata
    check_array_result("XYMode2", x, s1.y[::2])
    check_array_result("XYMode2", y, s2.y)
    check_array_result("XYMode2", x**2 + y**2, np.ones_like(x))


if __name__ == "__main__":
    test_signal_calibration()
    test_signal_swap_axes()
    test_to_polar()
    test_to_cartesian()
    test_signal_to_polar()
    test_signal_to_cartesian()
    test_signal_reverse_x()
    test_signal_normalize()
    test_signal_clip()
    test_signal_convolution()
    test_signal_derivative()
    test_signal_integral()
    test_signal_offset_correction()
    test_signal_gaussian_filter()
    test_signal_moving_average()
    test_signal_moving_median()
    test_signal_wiener()
    test_signal_resampling()
    test_signal_XY_mode()
