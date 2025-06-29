# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for signal analysis features
---------------------------------------

Features from the "Analysis" menu are covered by this test.
The "Analysis" menu contains functions to compute signal properties like
bandwidth, ENOB, etc.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import pytest

import sigima_.computation.signal as sigima_signal
import sigima_.obj
import sigima_.param
from cdl.tests.data import get_test_signal
from sigima_.tests.helpers import check_scalar_result


@pytest.mark.validation
def test_signal_bandwidth_3db() -> None:
    """Validation test for the bandwidth computation."""
    obj = get_test_signal("bandwidth.txt")
    res = sigima_signal.bandwidth_3db(obj)
    assert res is not None, "Bandwidth computation failed"
    df = res.to_dataframe()
    check_scalar_result("Bandwitdh@-3dB", df.L[0], 39.0, rtol=0.001)


@pytest.mark.validation
def test_dynamic_parameters() -> None:
    """Validation test for dynamic parameters computation."""
    obj = get_test_signal("dynamic_parameters.txt")
    param = sigima_.param.DynamicParam.create(full_scale=1.0)
    res = sigima_signal.dynamic_parameters(obj, param)
    assert res is not None, "Dynamic parameters computation failed"
    df = res.to_dataframe()
    check_scalar_result("ENOB", df.ENOB[0], 5.1, rtol=0.001)
    check_scalar_result("SINAD", df.SINAD[0], 32.49, rtol=0.001)
    check_scalar_result("THD", df.THD[0], -30.18, rtol=0.001)
    check_scalar_result("SFDR", df.SFDR[0], 34.03, rtol=0.001)
    check_scalar_result("Freq", df.Freq[0], 49998377.464, rtol=0.001)
    check_scalar_result("SNR", df.SNR[0], 101.52, rtol=0.001)


@pytest.mark.validation
def test_signal_sampling_rate_period() -> None:
    """Validation test for the sampling rate and period computation."""
    obj = get_test_signal("dynamic_parameters.txt")
    res = sigima_signal.sampling_rate_period(obj)
    assert res is not None, "Sampling rate and period computation failed"
    df = res.to_dataframe()
    check_scalar_result("Sampling rate", df["fs"][0], 1.0e10, rtol=0.001)
    check_scalar_result("Period", df["T"][0], 1.0e-10, rtol=0.001)


@pytest.mark.validation
def test_signal_contrast() -> None:
    """Validation test for the contrast computation."""
    obj = get_test_signal("fw1e2.txt")
    res = sigima_signal.contrast(obj)
    assert res is not None, "Contrast computation failed"
    df = res.to_dataframe()
    check_scalar_result("Contrast", df.contrast[0], 0.825, rtol=0.001)


@pytest.mark.validation
def test_signal_x_at_minmax() -> None:
    """Validation test for the x value at min/max computation."""
    obj = get_test_signal("fw1e2.txt")
    res = sigima_signal.x_at_minmax(obj)
    assert res is not None, "X at min/max computation failed"
    df = res.to_dataframe()
    check_scalar_result("X@Ymin", df["X@Ymin"][0], 0.803, rtol=0.001)
    check_scalar_result("X@Ymax", df["X@Ymax"][0], 5.184, rtol=0.001)


@pytest.mark.validation
def test_signal_x_at_y() -> None:
    """Validation test for the abscissa finding computation."""
    newparam = sigima_.obj.NewSignalParam.create(stype=sigima_.obj.SignalTypes.STEP)
    extra_param = sigima_.obj.StepParam()
    obj = sigima_.obj.create_signal_from_param(newparam, extra_param=extra_param)
    if obj is None:
        raise ValueError("Failed to create test signal")
    param = sigima_signal.OrdinateParam.create(y=0.5)
    res = sigima_signal.x_at_y(obj, param)
    assert res is not None, "X at Y computation failed"
    df = res.to_dataframe()
    check_scalar_result("x|y=0.5", df["x"][0], 0.0)


@pytest.mark.validation
def test_signal_y_at_x() -> None:
    """Validation test for the ordinate finding computation."""
    newparam = sigima_.obj.NewSignalParam.create(
        stype=sigima_.obj.SignalTypes.TRIANGLE, xmin=0.0, xmax=10.0, size=101
    )
    extra_param = sigima_.obj.PeriodicParam()
    obj = sigima_.obj.create_signal_from_param(newparam, extra_param=extra_param)
    if obj is None:
        raise ValueError("Failed to create test signal")
    param = sigima_signal.AbscissaParam.create(x=2.5)
    res = sigima_signal.y_at_x(obj, param)
    assert res is not None, "Y at X computation failed"
    df = res.to_dataframe()
    check_scalar_result("y|x=0.5", df["y"][0], 1.0)


if __name__ == "__main__":
    test_signal_bandwidth_3db()
    test_dynamic_parameters()
    test_signal_sampling_rate_period()
    test_signal_contrast()
    test_signal_x_at_minmax()
    test_signal_x_at_y()
    test_signal_y_at_x()
