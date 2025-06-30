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

import warnings

import numpy as np
import pytest

import sigima_.computation.signal as sigima_signal
import sigima_.obj
import sigima_.param as sigima_param
import sigima_.tests.data as ctd
from sigima_.tests.helpers import check_array_result


def __create_two_signals() -> tuple[sigima_.obj.SignalObj, sigima_.obj.SignalObj]:
    """Create two signals for testing."""
    s1 = ctd.create_periodic_signal(
        sigima_.obj.SignalTypes.COSINUS, freq=50.0, size=100
    )
    s2 = ctd.create_periodic_signal(sigima_.obj.SignalTypes.SINUS, freq=25.0, size=100)
    return s1, s2


def __create_n_signals(n: int = 100) -> list[sigima_.obj.SignalObj]:
    """Create a list of N different signals for testing."""
    signals = []
    for i in range(n):
        s = ctd.create_periodic_signal(
            sigima_.obj.SignalTypes.COSINUS,
            freq=50.0 + i,
            size=100,
            a=(i + 1) * 0.1,
        )
        signals.append(s)
    return signals


def __create_one_signal_and_constant() -> tuple[
    sigima_.obj.SignalObj, sigima_param.ConstantParam
]:
    """Create one signal and a constant for testing."""
    s1 = ctd.create_periodic_signal(
        sigima_.obj.SignalTypes.COSINUS, freq=50.0, size=100
    )
    param = sigima_param.ConstantParam.create(value=-np.pi)
    return s1, param


@pytest.mark.validation
def test_signal_addition() -> None:
    """Signal addition test."""
    slist = __create_n_signals()
    n = len(slist)
    s3 = sigima_signal.addition(slist)
    res = s3.y
    exp = np.zeros_like(s3.y)
    for s in slist:
        exp += s.y
    check_array_result(f"Addition of {n} signals", res, exp)


@pytest.mark.validation
def test_signal_average() -> None:
    """Signal average test."""
    slist = __create_n_signals()
    n = len(slist)
    s3 = sigima_signal.average(slist)
    res = s3.y
    exp = np.zeros_like(s3.y)
    for s in slist:
        exp += s.y
    exp /= n
    check_array_result(f"Average of {n} signals", res, exp)


@pytest.mark.validation
def test_signal_product() -> None:
    """Signal multiplication test."""
    slist = __create_n_signals()
    n = len(slist)
    s3 = sigima_signal.product(slist)
    res = s3.y
    exp = np.ones_like(s3.y)
    for s in slist:
        exp *= s.y
    check_array_result(f"Product of {n} signals", res, exp)


@pytest.mark.validation
def test_signal_difference() -> None:
    """Signal difference test."""
    s1, s2 = __create_two_signals()
    s3 = sigima_signal.difference(s1, s2)
    check_array_result("Signal difference", s3.y, s1.y - s2.y)


@pytest.mark.validation
def test_signal_quadratic_difference() -> None:
    """Signal quadratic difference validation test."""
    s1, s2 = __create_two_signals()
    s3 = sigima_signal.quadratic_difference(s1, s2)
    check_array_result("Signal quadratic difference", s3.y, (s1.y - s2.y) / np.sqrt(2))


@pytest.mark.validation
def test_signal_division() -> None:
    """Signal division test."""
    s1, s2 = __create_two_signals()
    s3 = sigima_signal.division(s1, s2)
    check_array_result("Signal division", s3.y, s1.y / s2.y)


@pytest.mark.validation
def test_signal_addition_constant() -> None:
    """Signal addition with constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima_signal.addition_constant(s1, param)
    check_array_result("Signal addition with constant", s2.y, s1.y + param.value)


@pytest.mark.validation
def test_signal_product_constant() -> None:
    """Signal multiplication by constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima_signal.product_constant(s1, param)
    check_array_result("Signal multiplication by constant", s2.y, s1.y * param.value)


@pytest.mark.validation
def test_signal_difference_constant() -> None:
    """Signal difference with constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima_signal.difference_constant(s1, param)
    check_array_result("Signal difference with constant", s2.y, s1.y - param.value)


@pytest.mark.validation
def test_signal_division_constant() -> None:
    """Signal division by constant test."""
    s1, param = __create_one_signal_and_constant()
    s2 = sigima_signal.division_constant(s1, param)
    check_array_result("Signal division by constant", s2.y, s1.y / param.value)


@pytest.mark.validation
def test_signal_inverse() -> None:
    """Signal inversion validation test."""
    s1 = __create_two_signals()[0]
    inv_signal = sigima_signal.inverse(s1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        exp = 1.0 / s1.y
        exp[np.isinf(exp)] = np.nan
    check_array_result("Signal inverse", inv_signal.y, exp)


@pytest.mark.validation
def test_signal_absolute() -> None:
    """Absolute value validation test."""
    s1 = __create_two_signals()[0]
    abs_signal = sigima_signal.absolute(s1)
    check_array_result("Absolute value", abs_signal.y, np.abs(s1.y))


@pytest.mark.validation
def test_signal_real() -> None:
    """Real part validation test."""
    s1 = __create_two_signals()[0]
    re_signal = sigima_signal.real(s1)
    check_array_result("Real part", re_signal.y, np.real(s1.y))


@pytest.mark.validation
def test_signal_imag() -> None:
    """Imaginary part validation test."""
    s1 = __create_two_signals()[0]
    im_signal = sigima_signal.imag(s1)
    check_array_result("Imaginary part", im_signal.y, np.imag(s1.y))


@pytest.mark.validation
def test_signal_astype() -> None:
    """Data type conversion validation test."""
    s1 = __create_two_signals()[0]
    for dtype_str in sigima_.obj.SignalObj.get_valid_dtypenames():
        p = sigima_param.DataTypeSParam.create(dtype_str=dtype_str)
        astype_signal = sigima_signal.astype(s1, p)
        assert astype_signal.y.dtype == np.dtype(dtype_str)


@pytest.mark.validation
def test_signal_exp() -> None:
    """Exponential validation test."""
    s1 = __create_two_signals()[0]
    exp_signal = sigima_signal.exp(s1)
    check_array_result("Exponential", exp_signal.y, np.exp(s1.y))


@pytest.mark.validation
def test_signal_log10() -> None:
    """Logarithm base 10 validation test."""
    s1 = __create_two_signals()[0]
    log10_signal = sigima_signal.log10(sigima_signal.exp(s1))
    check_array_result("Logarithm base 10", log10_signal.y, np.log10(np.exp(s1.y)))


@pytest.mark.validation
def test_signal_sqrt() -> None:
    """Square root validation test."""
    s1 = ctd.get_test_signal("paracetamol.txt")
    sqrt_signal = sigima_signal.sqrt(s1)
    check_array_result("Square root", sqrt_signal.y, np.sqrt(s1.y))


@pytest.mark.validation
def test_signal_power() -> None:
    """Power validation test."""
    s1 = ctd.get_test_signal("paracetamol.txt")
    p = sigima_param.PowerParam.create(power=2.0)
    power_signal = sigima_signal.power(s1, p)
    check_array_result("Power", power_signal.y, s1.y**p.power)


@pytest.mark.validation
def test_signal_arithmetic() -> None:
    """Arithmetic operations validation test."""
    s1, s2 = __create_two_signals()
    p = sigima_param.ArithmeticParam.create()
    for operator in p.operators:
        p.operator = operator
        for factor in (0.0, 1.0, 2.0):
            p.factor = factor
            for constant in (0.0, 1.0, 2.0):
                p.constant = constant
                s3 = sigima_signal.arithmetic(s1, s2, p)
                if operator == "+":
                    exp = s1.y + s2.y
                elif operator == "×":
                    exp = s1.y * s2.y
                elif operator == "-":
                    exp = s1.y - s2.y
                elif operator == "/":
                    exp = s1.y / s2.y
                exp = exp * factor + constant
                check_array_result(f"Arithmetic [{p.get_operation()}]", s3.y, exp)


if __name__ == "__main__":
    test_signal_addition()
    test_signal_average()
    test_signal_product()
    test_signal_difference()
    test_signal_quadratic_difference()
    test_signal_division()
    test_signal_addition_constant()
    test_signal_product_constant()
    test_signal_difference_constant()
    test_signal_division_constant()
    test_signal_inverse()
    test_signal_absolute()
    test_signal_real()
    test_signal_imag()
    test_signal_astype()
    test_signal_exp()
    test_signal_log10()
    test_signal_sqrt()
    test_signal_power()
    test_signal_arithmetic()
