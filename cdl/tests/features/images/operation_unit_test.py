# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for image operations
-------------------------------

Features from the "Operations" menu are covered by this test.
The "Operations" menu contains basic operations on images, such as
addition, multiplication, division, and more.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import warnings
from typing import Generator

import numpy as np
import pytest
import scipy.ndimage as spi
from guidata.qthelpers import qt_app_context

import cdl.computation.image as cpi
import cdl.obj
import cdl.param
from cdl.algorithms.datatypes import is_integer_dtype
from cdl.env import execenv
from cdl.tests.data import create_noisygauss_image
from cdl.utils.tests import check_array_result
from cdl.utils.vistools import view_images_side_by_side


def __iterate_images() -> Generator[cdl.obj.ImageObj, None, None]:
    """Iterate over all possible image types for testing."""
    size = 128
    for dtype in cdl.obj.ImageDatatypes:
        param = cdl.obj.NewImageParam.create(dtype=dtype, height=size, width=size)
        yield create_noisygauss_image(param, level=0.0)


def __iterate_image_couples() -> (
    Generator[tuple[cdl.obj.ImageObj, cdl.obj.ImageObj], None, None]
):
    """Iterate over all possible image couples for testing."""
    size = 128
    for dtype1 in cdl.obj.ImageDatatypes:
        param1 = cdl.obj.NewImageParam.create(dtype=dtype1, height=size, width=size)
        ima1 = create_noisygauss_image(param1, level=0.0)
        for dtype2 in cdl.obj.ImageDatatypes:
            param2 = cdl.obj.NewImageParam.create(dtype=dtype2, height=size, width=size)
            ima2 = create_noisygauss_image(param2, level=0.0)
            yield ima1, ima2


@pytest.mark.validation
def test_image_addition() -> None:
    """Image addition test."""
    execenv.print("*** Testing image addition:")
    for ima1, ima2 in __iterate_image_couples():
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  {dtype1} += {dtype2}: ", end="")
        exp = ima1.data.astype(float) + ima2.data.astype(float)
        cpi.compute_addition(ima2, ima1)
        check_array_result("Image addition", ima2.data, exp)


@pytest.mark.validation
def test_image_difference() -> None:
    """Image difference test."""
    execenv.print("*** Testing image difference:")
    for ima1, ima2 in __iterate_image_couples():
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  {dtype1} -= {dtype2}: ", end="")
        exp = ima1.data.astype(float) - ima2.data.astype(float)
        ima3 = cpi.compute_difference(ima1, ima2)
        check_array_result("Image difference", ima3.data, exp)


@pytest.mark.validation
def test_image_quadratic_difference() -> None:
    """Quadratic difference test."""
    execenv.print("*** Testing quadratic difference:")
    for ima1, ima2 in __iterate_image_couples():
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  ({dtype1} - {dtype2})/√2: ", end="")
        exp = (ima1.data.astype(float) - ima2.data.astype(float)) / np.sqrt(2)
        ima3 = cpi.compute_quadratic_difference(ima1, ima2)
        check_array_result("Image quadratic difference", ima3.data, exp)


@pytest.mark.validation
def test_image_product() -> None:
    """Image multiplication test."""
    execenv.print("*** Testing image multiplication:")
    for ima1, ima2 in __iterate_image_couples():
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  {dtype1} *= {dtype2}: ", end="")
        exp = ima1.data.astype(float) * ima2.data.astype(float)
        cpi.compute_product(ima2, ima1)
        check_array_result("Image multiplication", ima2.data, exp)


@pytest.mark.validation
def test_image_division() -> None:
    """Image division test."""
    execenv.print("*** Testing image division:")
    for ima1, ima2 in __iterate_image_couples():
        ima2.data = np.ones_like(ima2.data)
        dtype1, dtype2 = ima1.data.dtype, ima2.data.dtype
        execenv.print(f"  {dtype1} /= {dtype2}: ", end="")
        exp = ima1.data.astype(float) / ima2.data.astype(float)
        ima3 = cpi.compute_division(ima1, ima2)
        if not np.allclose(ima3.data, exp):
            with qt_app_context():
                view_images_side_by_side(
                    [ima1.data, ima2.data, ima3.data], ["ima1", "ima2", "ima3"]
                )
        check_array_result("Image division", ima3.data, exp)


def __constparam(value: float) -> cdl.param.ConstantParam:
    """Create a constant parameter."""
    return cdl.param.ConstantParam.create(value=value)


def __iterate_image_with_constant() -> (
    Generator[tuple[cdl.obj.ImageObj, cdl.param.ConstantParam], None, None]
):
    """Iterate over all possible image and constant couples for testing."""
    size = 128
    for dtype in cdl.obj.ImageDatatypes:
        param = cdl.obj.NewImageParam.create(dtype=dtype, height=size, width=size)
        ima = create_noisygauss_image(param, level=0.0)
        for value in (-1.0, 3.14, 5):
            p = __constparam(value)
            yield ima, p


@pytest.mark.validation
def test_image_addition_constant() -> None:
    """Image addition with constant test."""
    execenv.print("*** Testing image addition with constant:")
    for ima1, p in __iterate_image_with_constant():
        dtype1 = ima1.data.dtype
        execenv.print(f"  {dtype1} += constant ({p.value}): ", end="")
        expvalue = np.array(p.value).astype(dtype=dtype1)
        exp = ima1.data.astype(float) + expvalue
        ima2 = cpi.compute_addition_constant(ima1, p)
        check_array_result(f"Image + constant ({p.value})", ima2.data, exp)


@pytest.mark.validation
def test_image_difference_constant() -> None:
    """Image difference with constant test."""
    execenv.print("*** Testing image difference with constant:")
    for ima1, p in __iterate_image_with_constant():
        dtype1 = ima1.data.dtype
        execenv.print(f"  {dtype1} -= constant ({p.value}): ", end="")
        expvalue = np.array(p.value).astype(dtype=dtype1)
        exp = ima1.data.astype(float) - expvalue
        ima2 = cpi.compute_difference_constant(ima1, p)
        check_array_result(f"Image - constant ({p.value})", ima2.data, exp)


@pytest.mark.validation
def test_image_product_constant() -> None:
    """Image multiplication by constant test."""
    execenv.print("*** Testing image multiplication by constant:")
    for ima1, p in __iterate_image_with_constant():
        dtype1 = ima1.data.dtype
        execenv.print(f"  {dtype1} *= constant ({p.value}): ", end="")
        exp = ima1.data.astype(float) * p.value
        ima2 = cpi.compute_product_constant(ima1, p)
        check_array_result(f"Image x constant ({p.value})", ima2.data, exp)


@pytest.mark.validation
def test_image_division_constant() -> None:
    """Image division by constant test."""
    execenv.print("*** Testing image division by constant:")
    for ima1, p in __iterate_image_with_constant():
        dtype1 = ima1.data.dtype
        execenv.print(f"  {dtype1} /= constant ({p.value}): ", end="")
        exp = ima1.data.astype(float) / p.value
        ima2 = cpi.compute_division_constant(ima1, p)
        check_array_result(f"Image / constant ({p.value})", ima2.data, exp)


@pytest.mark.validation
def test_image_arithmetic() -> None:
    """Image arithmetic test."""
    execenv.print("*** Testing image arithmetic:")
    for ima1, ima2 in __iterate_image_couples():
        dtype1 = ima1.data.dtype
        p = cdl.param.ArithmeticParam.create()
        for o in p.operators:
            p.operator = o
            for a in (0.0, 1.0, 2.0):
                p.factor = a
                for b in (0.0, 1.0, 2.0):
                    p.constant = b
                    ima2.data = np.clip(ima2.data, 1, None)  # Avoid division by zero
                    ima3 = cpi.compute_arithmetic(ima1, ima2, p)
                    if o in ("×", "/") and a == 0.0:
                        exp = np.ones_like(ima1.data) * b
                    elif o == "+":
                        exp = np.add(ima1.data, ima2.data, dtype=float) * a + b
                    elif o == "×":
                        exp = np.multiply(ima1.data, ima2.data, dtype=float) * a + b
                    elif o == "-":
                        exp = np.subtract(ima1.data, ima2.data, dtype=float) * a + b
                    elif o == "/":
                        exp = np.divide(ima1.data, ima2.data, dtype=float) * a + b
                    if p.restore_dtype:
                        if is_integer_dtype(dtype1):
                            iinfo1 = np.iinfo(dtype1)
                            exp = np.clip(exp, iinfo1.min, iinfo1.max)
                        exp = exp.astype(dtype1)
                    check_array_result(
                        f"Arithmetic [{p.get_operation()}]", ima3.data, exp
                    )


@pytest.mark.validation
def test_image_inverse() -> None:
    """Image inverse test."""
    execenv.print("*** Testing image inverse:")
    for ima1 in __iterate_images():
        execenv.print(f"  1/({ima1.data.dtype}): ", end="")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            exp = np.reciprocal(ima1.data, dtype=float)
            exp[np.isinf(exp)] = np.nan
        ima2 = cpi.compute_inverse(ima1)
        check_array_result("Image inverse", ima2.data, exp)


@pytest.mark.validation
def test_image_abs() -> None:
    """Image absolute value test."""
    execenv.print("*** Testing image absolute value:")
    for ima1 in __iterate_images():
        execenv.print(f"  abs({ima1.data.dtype}): ", end="")
        exp = np.abs(ima1.data)
        ima2 = cpi.compute_abs(ima1)
        check_array_result("Image abs", ima2.data, exp)


@pytest.mark.validation
def test_image_re() -> None:
    """Image real part test."""
    execenv.print("*** Testing image real part:")
    for ima1 in __iterate_images():
        execenv.print(f"  re({ima1.data.dtype}): ", end="")
        exp = np.real(ima1.data)
        ima2 = cpi.compute_re(ima1)
        check_array_result("Image re", ima2.data, exp)


@pytest.mark.validation
def test_image_im() -> None:
    """Image imaginary part test."""
    execenv.print("*** Testing image imaginary part:")
    for ima1 in __iterate_images():
        execenv.print(f"  im({ima1.data.dtype}): ", end="")
        exp = np.imag(ima1.data)
        ima2 = cpi.compute_im(ima1)
        check_array_result("Image im", ima2.data, exp)


def __get_numpy_info(dtype: np.dtype) -> np.generic:
    """Get numpy info for a given data type."""
    if is_integer_dtype(dtype):
        return np.iinfo(dtype)
    return np.finfo(dtype)


@pytest.mark.validation
def test_image_astype() -> None:
    """Image type conversion test."""
    execenv.print("*** Testing image type conversion:")
    for ima1 in __iterate_images():
        for dtype_str in cpi.VALID_DTYPES_STRLIST:
            dtype1_str = str(ima1.data.dtype)
            execenv.print(f"  {dtype1_str} -> {dtype_str}: ", end="")
            dtype_exp = np.dtype(dtype_str)
            info_exp = __get_numpy_info(dtype_exp)
            info_ima1 = __get_numpy_info(ima1.data.dtype)
            if info_exp.min < info_ima1.min or info_exp.max > info_ima1.max:
                continue
            exp = np.clip(ima1.data, info_exp.min, info_exp.max).astype(dtype_exp)
            p = cdl.param.DataTypeIParam.create(dtype_str=dtype_str)
            ima2 = cpi.compute_astype(ima1, p)
            check_array_result(
                f"Image astype({dtype1_str}->{dtype_str})", ima2.data, exp
            )


@pytest.mark.validation
def test_image_exp() -> None:
    """Image exponential test."""
    execenv.print("*** Testing image exponential:")
    with np.errstate(over="ignore"):
        for ima1 in __iterate_images():
            execenv.print(f"  exp({ima1.data.dtype}): ", end="")
            exp = np.exp(ima1.data)
            ima2 = cpi.compute_exp(ima1)
            check_array_result("Image exp", ima2.data, exp)


@pytest.mark.validation
def test_image_log10() -> None:
    """Image base-10 logarithm test."""
    execenv.print("*** Testing image base-10 logarithm:")
    with np.errstate(over="ignore"):
        for ima1 in __iterate_images():
            execenv.print(f"  log10({ima1.data.dtype}): ", end="")
            exp = np.log10(np.exp(ima1.data))
            ima2 = cpi.compute_log10(cpi.compute_exp(ima1))
            check_array_result("Image log10", ima2.data, exp)


@pytest.mark.validation
def test_image_logp1() -> None:
    """Image log(1+n) test."""
    execenv.print("*** Testing image log(1+n):")
    with np.errstate(over="ignore"):
        for ima1 in __iterate_images():
            execenv.print(f"  log1p({ima1.data.dtype}): ", end="")
            p = cdl.param.LogP1Param.create(n=2)
            exp = np.log10(ima1.data + p.n)
            ima2 = cpi.compute_logp1(ima1, p)
            check_array_result("Image log1p", ima2.data, exp)


def __generic_flip_check(compfunc: callable, expfunc: callable) -> None:
    """Generic flip check function."""
    execenv.print(f"*** Testing image flip: {compfunc.__name__}")
    for ima1 in __iterate_images():
        execenv.print(f"  {compfunc.__name__}({ima1.data.dtype}): ", end="")
        ima2: cdl.obj.ImageObj = compfunc(ima1)
        check_array_result("Image flip", ima2.data, expfunc(ima1.data))


@pytest.mark.validation
def test_image_fliph() -> None:
    """Image horizontal flip test."""
    __generic_flip_check(cpi.compute_fliph, np.fliplr)


@pytest.mark.validation
def test_image_flipd() -> None:
    """Image diagonal flip test."""
    __generic_flip_check(cpi.compute_swap_axes, np.transpose)


@pytest.mark.validation
def test_image_flipv() -> None:
    """Image vertical flip test."""
    __generic_flip_check(cpi.compute_flipv, np.flipud)


def __generic_rotate_check(angle: int) -> None:
    """Generic rotate check function."""
    execenv.print(f"*** Testing image {angle}° rotation:")
    for ima1 in __iterate_images():
        execenv.print(f"  rotate{angle}({ima1.data.dtype}): ", end="")
        ima2 = getattr(cpi, f"compute_rotate{angle}")(ima1)
        check_array_result(
            f"Image rotate{angle}", ima2.data, np.rot90(ima1.data, k=angle // 90)
        )


@pytest.mark.validation
def test_image_rotate90() -> None:
    """Image 90° rotation test."""
    __generic_rotate_check(90)


@pytest.mark.validation
def test_image_rotate270() -> None:
    """Image 270° rotation test."""
    __generic_rotate_check(270)


@pytest.mark.validation
def test_image_rotate() -> None:
    """Image rotation test."""
    execenv.print("*** Testing image rotation:")
    for ima1 in __iterate_images():
        for angle in (30, 45, 60, 120):
            execenv.print(f"  rotate{angle}({ima1.data.dtype}): ", end="")
            ima2 = cpi.compute_rotate(ima1, cdl.param.RotateParam.create(angle=angle))
            exp = spi.rotate(ima1.data, angle, reshape=False)
            check_array_result(f"Image rotate{angle}", ima2.data, exp)


if __name__ == "__main__":
    test_image_addition()
    test_image_product()
    test_image_division()
    test_image_difference()
    test_image_quadratic_difference()
    test_image_addition_constant()
    test_image_product_constant()
    test_image_difference_constant()
    test_image_division_constant()
    test_image_arithmetic()
    test_image_inverse()
    test_image_abs()
    test_image_re()
    test_image_im()
    test_image_astype()
    test_image_exp()
    test_image_log10()
    test_image_logp1()
    test_image_fliph()
    test_image_flipv()
    test_image_rotate90()
    test_image_rotate270()
    test_image_rotate()
