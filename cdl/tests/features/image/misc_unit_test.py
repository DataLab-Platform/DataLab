# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Miscellaneous image tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
import pytest
from guidata.qthelpers import qt_app_context

import cdl.core.computation.image as cpi
import cdl.obj
import cdl.param
from cdl.env import execenv
from cdl.tests.data import create_noisygauss_image
from cdl.utils.vistools import view_images_side_by_side


@pytest.mark.validation
def test_image_addition() -> None:
    """Image addition test."""
    execenv.print("Testing image addition:")
    size = 128
    for dtype1 in cdl.obj.ImageDatatypes:
        param1 = cdl.obj.NewImageParam.create(dtype=dtype1, height=size, width=size)
        ima1 = create_noisygauss_image(param1, level=0.0)
        for dtype2 in cdl.obj.ImageDatatypes:
            execenv.print(f"  {dtype1.name} += {dtype2.name}: ", end="")
            param2 = cdl.obj.NewImageParam.create(dtype=dtype2, height=size, width=size)
            ima2 = create_noisygauss_image(param2, level=0.0)
            exp = ima1.data.copy()
            exp += np.array(ima2.data, dtype=ima1.data.dtype)
            cpi.compute_addition(ima1, ima2)
            res = ima1.data
            assert np.allclose(
                res, exp
            ), f"Image addition failed for dtype {dtype1.name} + {dtype2.name}"
            execenv.print("OK")


@pytest.mark.validation
def test_image_addition_constant() -> None:
    """Image addition with constant test."""
    execenv.print("Testing image addition with constant:")
    size = 128
    for dtype in cdl.obj.ImageDatatypes:
        execenv.print(f"  {dtype.name} += constant: ", end="")
        p = cdl.param.ConstantOperationParam.create(value=1.0)
        ima1 = create_noisygauss_image(
            cdl.obj.NewImageParam.create(dtype=dtype, height=size, width=size),
            level=0.0,
        )
        exp = ima1.data.copy()
        exp += np.array(p.value, dtype=ima1.data.dtype)
        ima2 = cpi.compute_addition_constant(ima1, p)
        res = ima2.data
        assert np.allclose(
            res, exp
        ), f"Image addition with constant failed for dtype {dtype.name}"
        execenv.print("OK")


@pytest.mark.validation
def test_image_difference() -> None:
    """Image difference test."""
    execenv.print("Testing image difference:")
    size = 128
    for dtype1 in cdl.obj.ImageDatatypes:
        param1 = cdl.obj.NewImageParam.create(dtype=dtype1, height=size, width=size)
        ima1 = create_noisygauss_image(param1, level=0.0)
        for dtype2 in cdl.obj.ImageDatatypes:
            execenv.print(f"  {dtype1.name} -= {dtype2.name}: ", end="")
            param2 = cdl.obj.NewImageParam.create(dtype=dtype2, height=size, width=size)
            ima2 = create_noisygauss_image(param2, level=0.0)
            exp = ima1.data - ima2.data
            ima3 = cpi.compute_difference(ima1, ima2)
            res = ima3.data
            assert np.allclose(
                res, exp
            ), f"Image difference failed for dtype {dtype1.name} - {dtype2.name}"
            execenv.print("OK")


@pytest.mark.validation
def test_image_difference_constant() -> None:
    """Image difference with constant test."""
    execenv.print("Testing image difference with constant:")
    size = 128
    for dtype in cdl.obj.ImageDatatypes:
        execenv.print(f"  {dtype.name} -= constant: ", end="")
        p = cdl.param.ConstantOperationParam.create(value=1.0)
        param1 = cdl.obj.NewImageParam.create(dtype=dtype, height=size, width=size)
        ima1 = create_noisygauss_image(param1, level=0.0)
        exp = ima1.data.copy()
        exp -= np.array(p.value, dtype=ima1.data.dtype)
        ima2 = cpi.compute_difference_constant(ima1, p)
        res = ima2.data
        assert np.allclose(
            res, exp
        ), f"Image difference with constant failed for dtype {dtype.name}"
        execenv.print("OK")


@pytest.mark.validation
def test_image_product() -> None:
    """Image multiplication test."""
    execenv.print("Testing image multiplication:")
    size = 128
    for dtype1 in cdl.obj.ImageDatatypes:
        param1 = cdl.obj.NewImageParam.create(dtype=dtype1, height=size, width=size)
        ima1 = create_noisygauss_image(param1, level=0.0)
        for dtype2 in cdl.obj.ImageDatatypes:
            execenv.print(f"  {dtype1.name} *= {dtype2.name}: ", end="")
            param2 = cdl.obj.NewImageParam.create(dtype=dtype2, height=size, width=size)
            ima2 = create_noisygauss_image(param2, level=0.0)
            exp = ima1.data.copy()
            exp *= np.array(ima2.data, dtype=ima1.data.dtype)
            cpi.compute_product(ima1, ima2)
            res = ima1.data
            assert np.allclose(
                res, exp
            ), f"Image multiplication failed for dtype {dtype1.name} * {dtype2.name}"
            execenv.print("OK")


@pytest.mark.validation
def test_image_product_constant() -> None:
    """Image multiplication by constant test."""
    execenv.print("Testing image multiplication by constant:")
    size = 128
    for dtype in cdl.obj.ImageDatatypes:
        execenv.print(f"  {dtype.name} *= constant: ", end="")
        p = cdl.param.ConstantOperationParam.create(value=2.0)
        ima1 = create_noisygauss_image(
            cdl.obj.NewImageParam.create(dtype=dtype, height=size, width=size),
            level=0.0,
        )
        exp = ima1.data.copy()
        exp *= np.array(p.value, dtype=ima1.data.dtype)
        ima2 = cpi.compute_product_constant(ima1, p)
        res = ima2.data
        assert np.allclose(
            res, exp
        ), f"Image multiplication by constant failed for dtype {dtype.name}"
        execenv.print("OK")


@pytest.mark.validation
def test_image_division() -> None:
    """Image division test."""
    execenv.print("Testing image division:")
    size = 128
    for dtype1 in cdl.obj.ImageDatatypes:
        param1 = cdl.obj.NewImageParam.create(dtype=dtype1, height=size, width=size)
        ima1 = create_noisygauss_image(param1, level=0.0)
        for dtype2 in cdl.obj.ImageDatatypes:
            execenv.print(f"  {dtype1.name} /= {dtype2.name}: ", end="")
            param2 = cdl.obj.NewImageParam.create(
                itype=cdl.obj.ImageTypes.ZEROS, dtype=dtype2, height=size, width=size
            )
            ima2 = cdl.obj.create_image_from_param(param2)
            ima2.data += np.array(1.0, dtype=ima2.data.dtype)
            exp = ima1.data / np.array(ima2.data, dtype=ima1.data.dtype)
            ima3 = cpi.compute_division(ima1, ima2)
            res = ima3.data
            if not np.allclose(res, exp):
                with qt_app_context():
                    view_images_side_by_side(
                        [ima1.data, ima2.data, ima3.data], ["ima1", "ima2", "ima3"]
                    )
            assert np.allclose(
                res, exp
            ), f"Image division failed for dtype {dtype1.name} / {dtype2.name}"
            execenv.print("OK")


@pytest.mark.validation
def test_image_division_constant() -> None:
    """Image division by constant test."""
    execenv.print("Testing image division by constant:")
    size = 128
    for dtype in cdl.obj.ImageDatatypes:
        execenv.print(f"  {dtype.name} /= constant: ", end="")
        p = cdl.param.ConstantOperationParam.create(value=2.0)
        param1 = cdl.obj.NewImageParam.create(dtype=dtype, height=size, width=size)
        ima1 = create_noisygauss_image(param1, level=0.0)
        exp = np.array(ima1.data / p.value, dtype=ima1.data.dtype)
        ima2 = cpi.compute_division_constant(ima1, p)
        res = ima2.data
        assert np.allclose(
            res, exp
        ), f"Image division by constant failed for dtype {dtype.name}"
        execenv.print("OK")


if __name__ == "__main__":
    test_image_addition()
    test_image_addition_constant()
    test_image_product()
    test_image_product_constant()
    test_image_division()
    test_image_division_constant()
    test_image_difference()
    test_image_difference_constant()
