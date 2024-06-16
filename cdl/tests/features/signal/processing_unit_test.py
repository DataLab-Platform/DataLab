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
from guidata.qthelpers import qt_app_context

import cdl.computation.signal as cps
import cdl.obj
import cdl.param
from cdl.tests.data import get_test_signal
from cdl.utils.tests import check_array_result
from cdl.utils.vistools import view_curves


@pytest.mark.validation
def test_signal_calibration() -> None:
    """Validation test for the signal calibration processing."""
    src = get_test_signal("paracetamol.txt")
    param = cdl.param.XYCalibrateParam()

    # Test with a = 1 and b = 0: should do nothing
    param.a, param.b = 1.0, 0.0
    for axis, _taxis in param.axes:
        param.axis = axis
        dst = cps.compute_calibration(src, param)
        exp = src.xydata
        check_array_result("Calibration[identity]", dst.xydata, exp)

    # Testing with random values of a and b
    param.a, param.b = 0.5, 0.1
    for axis, _taxis in param.axes:
        param.axis = axis
        exp_x1, exp_y1 = src.xydata.copy()
        if axis == "x":
            exp_x1 = param.a * exp_x1 + param.b
        else:
            exp_y1 = param.a * exp_y1 + param.b
        dst = cps.compute_calibration(src, param)
        res_x1, res_y1 = dst.xydata
        title = f"Calibration[{axis},a={param.a},b={param.b}]"
        check_array_result(f"{title}.x", res_x1, exp_x1)
        check_array_result(f"{title}.y", res_y1, exp_y1)


@pytest.mark.validation
def test_signal_clip() -> None:
    """Validation test for the signal clipping processing."""
    src = get_test_signal("paracetamol.txt")
    param = cdl.param.ClipParam()

    for lower, upper in ((float("-inf"), float("inf")), (250.0, 500.0)):
        param.lower, param.upper = lower, upper
        dst = cps.compute_clip(src, param)
        exp = np.clip(src.data, param.lower, param.upper)
        check_array_result(f"Clip[{lower},{upper}]", dst.data, exp)


@pytest.mark.validation
def test_signal_convolution() -> None:
    """Validation test for the signal convolution processing."""
    src1 = get_test_signal("paracetamol.txt")
    snew = cdl.obj.new_signal_param("Gaussian", stype=cdl.obj.SignalTypes.GAUSS)
    addparam = cdl.obj.GaussLorentzVoigtParam.create(sigma=10.0)
    src2 = cdl.obj.create_signal_from_param(snew, addparam=addparam, edit=False)

    dst = cps.compute_convolution(src1, src2)
    exp = np.convolve(src1.data, src2.data, mode="same")
    check_array_result("Convolution", dst.data, exp)


@pytest.mark.validation
def test_signal_derivative() -> None:
    """Validation test for the signal derivative processing."""
    src = get_test_signal("paracetamol.txt")

    dst = cps.compute_derivative(src)

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
    dst = cps.compute_integral(cps.compute_derivative(src))
    # The integral of the derivative should be the original signal, up to a constant:
    dst.y += src.y[0]

    check_array_result("Integral[Derivative]", dst.y, src.y, atol=0.05)

    dst = cps.compute_integral(src)

    x, y = src.xydata

    # Compute the integral using a simple trapezoidal rule:
    exp = np.zeros_like(y)
    exp[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    exp[0] = exp[1]

    check_array_result("Integral", dst.y, exp, atol=0.05)


if __name__ == "__main__":
    test_signal_calibration()
    test_signal_clip()
    test_signal_convolution()
    test_signal_derivative()
    test_signal_integral()
