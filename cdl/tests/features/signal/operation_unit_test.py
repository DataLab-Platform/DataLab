# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for signal operations
--------------------------------

Features from the "Operations" menu are covered by this test.
The "Operations" menu contains basic operations on signals, such as
addition, multiplication, division, and more.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest

import cdl.computation.signal as cps
import cdl.obj
import cdl.param
import cdl.tests.data as ctd
from cdl.utils.tests import check_array_result


def __create_two_signals() -> tuple[cdl.obj.SignalObj, cdl.obj.SignalObj]:
    """Create two signals for testing."""
    s1 = ctd.create_periodic_signal(cdl.obj.SignalTypes.COSINUS, freq=50.0, size=100)
    s2 = ctd.create_periodic_signal(cdl.obj.SignalTypes.SINUS, freq=25.0, size=100)
    return s1, s2


def __create_one_signal_and_constant() -> (
    tuple[cdl.obj.SignalObj, cdl.param.ConstantOperationParam]
):
    """Create one signal and a constant for testing."""
    s1 = ctd.create_periodic_signal(cdl.obj.SignalTypes.COSINUS, freq=50.0, size=100)
    param = cdl.param.ConstantOperationParam.create(value=-np.pi)
    return s1, param


@pytest.mark.validation
def test_signal_addition() -> None:
    """Signal addition test."""
    s1, s2 = __create_two_signals()
    exp = s1.y + s2.y
    cps.compute_addition(s1, s2)
    res = s1.y
    check_array_result("Signal addition", res, exp)


@pytest.mark.validation
def test_signal_product() -> None:
    """Signal multiplication test."""
    s1, s2 = __create_two_signals()
    exp = s1.y * s2.y
    cps.compute_product(s1, s2)
    res = s1.y
    check_array_result("Signal multiplication", res, exp)


@pytest.mark.validation
def test_signal_difference() -> None:
    """Signal difference test."""
    s1, s2 = __create_two_signals()
    s3 = cps.compute_difference(s1, s2)
    check_array_result("Signal difference", s3.y, s1.y - s2.y)


@pytest.mark.validation
def test_signal_quadratic_difference() -> None:
    """Signal quadratic difference validation test."""
    s1, s2 = __create_two_signals()
    s3 = cps.compute_quadratic_difference(s1, s2)
    check_array_result("Signal quadratic difference", s3.y, (s1.y - s2.y) / np.sqrt(2))


@pytest.mark.validation
def test_signal_division() -> None:
    """Signal division test."""
    s1, s2 = __create_two_signals()
    s3 = cps.compute_division(s1, s2)
    check_array_result("Signal division", s3.y, s1.y / s2.y)


@pytest.mark.validation
def test_signal_addition_constant() -> None:
    """Signal addition with constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = cps.compute_addition_constant(s1, param)
    check_array_result("Signal addition with constant", s2.y, s1.y + param.value)


@pytest.mark.validation
def test_signal_product_constant() -> None:
    """Signal multiplication by constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = cps.compute_product_constant(s1, param)
    check_array_result("Signal multiplication by constant", s2.y, s1.y * param.value)


@pytest.mark.validation
def test_signal_difference_constant() -> None:
    """Signal difference with constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = cps.compute_difference_constant(s1, param)
    check_array_result("Signal difference with constant", s2.y, s1.y - param.value)


@pytest.mark.validation
def test_signal_division_constant() -> None:
    """Signal division by constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = cps.compute_division_constant(s1, param)
    check_array_result("Signal division by constant", s2.y, s1.y / param.value)


@pytest.mark.validation
def test_signal_abs() -> None:
    """Absolute value validation test."""
    s1 = __create_two_signals()[0]
    abs_signal = cps.compute_abs(s1)
    check_array_result("Absolute value", abs_signal.y, np.abs(s1.y))


@pytest.mark.validation
def test_signal_re() -> None:
    """Real part validation test."""
    s1 = __create_two_signals()[0]
    re_signal = cps.compute_re(s1)
    check_array_result("Real part", re_signal.y, np.real(s1.y))


@pytest.mark.validation
def test_signal_im() -> None:
    """Imaginary part validation test."""
    s1 = __create_two_signals()[0]
    im_signal = cps.compute_im(s1)
    check_array_result("Imaginary part", im_signal.y, np.imag(s1.y))


@pytest.mark.validation
def test_signal_astype() -> None:
    """Data type conversion validation test."""
    s1 = __create_two_signals()[0]
    for dtype_str in cps.VALID_DTYPES_STRLIST:
        p = cdl.param.DataTypeSParam.create(dtype_str=dtype_str)
        astype_signal = cps.compute_astype(s1, p)
        assert astype_signal.y.dtype == np.dtype(dtype_str)


@pytest.mark.validation
def test_signal_exp() -> None:
    """Exponential validation test."""
    s1 = __create_two_signals()[0]
    exp_signal = cps.compute_exp(s1)
    check_array_result("Exponential", exp_signal.y, np.exp(s1.y))


@pytest.mark.validation
def test_signal_log10() -> None:
    """Logarithm base 10 validation test."""
    s1 = __create_two_signals()[0]
    log10_signal = cps.compute_log10(cps.compute_exp(s1))
    check_array_result("Logarithm base 10", log10_signal.y, np.log10(np.exp(s1.y)))


if __name__ == "__main__":
    test_signal_addition()
    test_signal_product()
    test_signal_difference()
    test_signal_quadratic_difference()
    test_signal_division()
    test_signal_addition_constant()
    test_signal_product_constant()
    test_signal_difference_constant()
    test_signal_division_constant()
    test_signal_abs()
    test_signal_re()
    test_signal_im()
    test_signal_astype()
    test_signal_exp()
    test_signal_log10()
