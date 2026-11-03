#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unit test for geometry transformation operations."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
import sigima.objects as sio
import sigima.params as sip
import sigima.proc.image as sipi
from sigima.tests import data as test_data

from datalab.adapters_metadata.geometry_adapter import GeometryAdapter
from datalab.gui.processor.geometry_postprocess import (
    GeometricTransformWrapper,
    apply_geometry_transform,
)
from datalab.tests import datalab_test_app_context


def __create_test_image() -> sio.ImageObj:
    """Create a test image with geometry results for testing."""
    # Geometry results and ROI below are defined in pixel coordinates
    param = sio.Gauss2DParam.create(
        height=600,
        width=600,
        x0=2.5,
        y0=7.5,
        title="Test image (with geometry results)",
        dtype=sio.ImageDatatypes.UINT16,
        physical_coords=False,
    )
    obj = sio.create_image_from_param(param)
    for geometry in test_data.generate_geometry_results():
        GeometryAdapter(geometry).add_to(obj)
    obj.roi = sio.create_image_roi("rectangle", [10, 10, 50, 400])
    return obj


def __get_geometries(obj: sio.ImageObj) -> list[sio.GeometryResult]:
    """Get geometries from an image object."""
    return [ga.result for ga in GeometryAdapter.iterate_from_obj(obj)]


def __get_expected_geometry_result(
    orig_geom: sio.GeometryResult, operation: str, src: sio.ImageObj
) -> sio.GeometryResult:
    """Get expected geometry result for a given operation.

    Args:
        orig_geom: Original geometry result
        operation: Operation name (rotate90, rotate270, etc.)
        src: Source image of the operation (uniform coordinates)

    Returns:
        Expected geometry result after transformation
    """
    # The image operations move the pixel centers: flips mirror the coordinates
    # about the center of the pixel grid (299.5, 299.5 for our 600x600 test image)
    nrows, ncols = src.data.shape
    xc = src.x0 + 0.5 * (ncols - 1) * src.dx
    yc = src.y0 + 0.5 * (nrows - 1) * src.dy

    if operation == "rotate90":
        # Counter-clockwise: (x, y) -> (y, 2 * xc - x)
        exp_geom = sipi.transformer.flipv(sipi.transformer.transpose(orig_geom), xc)
    elif operation == "rotate270":
        # Clockwise: (x, y) -> (2 * yc - y, x)
        exp_geom = sipi.transformer.fliph(sipi.transformer.transpose(orig_geom), yc)
    elif operation == "fliph":
        exp_geom = sipi.transformer.fliph(orig_geom, xc)
    elif operation == "flipv":
        exp_geom = sipi.transformer.flipv(orig_geom, yc)
    elif operation == "transpose":
        exp_geom = sipi.transformer.transpose(orig_geom)
    elif operation == "resize":
        # Resize changes pixel resolution but keeps physical coordinates unchanged
        exp_geom = orig_geom
    elif operation == "binning":
        # Binning changes pixel resolution but keeps physical coordinates unchanged
        exp_geom = orig_geom
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return exp_geom


def __validate_geometry_transformation(
    tr_geometries: list[sio.GeometryResult],
    or_geometries: list[sio.GeometryResult],
    operation: str,
    src: sio.ImageObj,
    test_context: str = "geometry",
) -> None:
    """Validate that geometry transformation was applied correctly.

    Args:
        tr_geometries: List of transformed geometry results
        or_geometries: List of original geometry results
        operation: Operation name (rotate90, rotate270, etc.)
        src: Source image of the operation
        test_context: Context string for error messages (e.g., "App", "Unit")
    """
    for i, tr_geom in enumerate(tr_geometries):
        original_geom = or_geometries[i]
        expected_geom = __get_expected_geometry_result(original_geom, operation, src)

        # Compare the actual transformation result with expected
        np.testing.assert_allclose(
            tr_geom.coords,
            expected_geom.coords,
            rtol=1e-10,
            err_msg=f"{test_context} geometry result {i} "
            f"({tr_geom.title}) coordinates do not match expected for {operation}. "
            f"Got: {tr_geom.coords}, Expected: {expected_geom.coords}",
        )

        # Verify other properties are preserved
        assert tr_geom.title == expected_geom.title, (
            f"Title should be preserved for geometry {i} in {operation}"
        )
        assert tr_geom.kind == expected_geom.kind, (
            f"Kind should be preserved for geometry {i} in {operation}"
        )
        np.testing.assert_array_equal(
            tr_geom.roi_indices,
            expected_geom.roi_indices,
            err_msg=f"ROI indices should be preserved for geometry {i} in {operation}",
        )


def __validate_basic_transformation(tr_obj: sio.ImageObj, or_obj: sio.ImageObj) -> None:
    """Validate basic transformation requirements.

    Args:
        tr_obj: The transformed image object
        or_obj: The original image object
    """
    assert tr_obj is not or_obj, "Transformation should create a new object"
    assert len(tr_obj.metadata) == len(or_obj.metadata), (
        "Number of geometry results should be preserved"
    )


def __validate_roi_transformation(
    tr_obj: sio.ImageObj,
    or_obj: sio.ImageObj,
    operation: str,
) -> None:
    """Validate that ROI is properly transformed.

    Args:
        tr_obj: The transformed image object
        or_obj: Original image object
        operation: Operation name for expected transformation
    """
    tr_roi = tr_obj.roi
    assert tr_roi is not None, "ROI should not be removed after transformation"
    assert len(tr_roi.single_rois) == len(or_obj.roi.single_rois), (
        "Number of ROI objects should be preserved"
    )
    # Validate that the ROI coordinates were properly transformed
    for i, roi in enumerate(tr_roi.single_rois):
        # Temporary transform the ROI into a geometry result:
        or_roi = or_obj.roi.get_single_roi(i)
        temp_geom = sio.GeometryResult(
            "temp_geom",
            sio.KindShape.RECTANGLE,
            coords=np.asarray([or_roi.get_physical_coords(or_obj)]),
        )
        expected_geom = __get_expected_geometry_result(temp_geom, operation, or_obj)
        exp_coords = expected_geom.coords[0]
        np.testing.assert_allclose(
            roi.get_physical_coords(tr_obj),
            exp_coords,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"ROI {i} coordinates do not match expected for {operation}.",
        )


def __validate_image_data_transformation(
    tr_obj: sio.ImageObj, or_obj: sio.ImageObj, operation: str
) -> None:
    """Validate that image data was properly transformed.

    Args:
        tr_obj: The transformed image object
        or_obj: Original image object
        operation: Operation name for expected transformation
    """
    if operation == "rotate90":
        expected_data = np.rot90(or_obj.data)
    elif operation == "rotate270":
        expected_data = np.rot90(or_obj.data, k=-1)
    elif operation == "fliph":
        expected_data = np.fliplr(or_obj.data)
    elif operation == "flipv":
        expected_data = np.flipud(or_obj.data)
    elif operation == "transpose":
        expected_data = np.transpose(or_obj.data)
    elif operation in ("resize", "binning"):
        # These operations are more complex and use Sigima's functions
        # We'll just verify the operation was applied (shape may change)
        return
    else:
        raise ValueError(f"Unknown operation for data validation: {operation}")

    np.testing.assert_array_equal(
        tr_obj.data,
        expected_data,
        err_msg=f"Image data not properly transformed for {operation}",
    )


# =============================================================================
# Unit Tests
# =============================================================================


def __create_unit_test(
    operation: str,
    param_creator: Callable[
        [], sip.BinningParam | sip.ResizeParam | sip.RotateParam
    ] = None,
) -> Callable[[], None]:
    """Create a unit test function for a geometry transformation operation.

    Args:
        operation: Operation name (e.g., "rotate90", "fliph")
        param_creator: Function to create parameter object, if needed

    Returns:
        Test function
    """

    def test_func() -> None:
        """Test geometry transformation at unit level."""
        obj = __create_test_image()
        or_geometries = __get_geometries(obj)  # Store original data for comparison

        # Apply transformation (same as DataLab processor does)
        param = param_creator() if param_creator else None
        compfunc = getattr(sipi, operation, None)
        if compfunc is None:
            raise ValueError(f"Unknown operation: {operation}")
        if param is None:
            tr_obj = compfunc(obj)
        else:
            tr_obj = compfunc(obj, param)

        # Apply the geometry transformation to the result
        # (skip geometry transformation for resize and binning as they preserve
        # physical coordinates)
        if operation not in ["resize", "binning"]:
            apply_geometry_transform(tr_obj, operation, src=obj)

        # Validate basic transformation requirements
        __validate_basic_transformation(tr_obj, obj)

        # Validate geometry transformation
        tr_geometries = __get_geometries(tr_obj)
        __validate_geometry_transformation(
            tr_geometries, or_geometries, operation, obj, "Unit"
        )

        # Validate ROI transformation (only for exact image operations)
        if operation not in ["resize", "binning"]:
            __validate_roi_transformation(tr_obj, obj, operation)

    # Set proper function name and docstring
    test_func.__name__ = f"test_geometry_transform_{operation}_unit"
    test_func.__doc__ = (
        f"Test geometry transformation for {operation} at unit level.\n\n"
        f"This test verifies that the DataLab geometry transformation properly "
        f"applies Sigima's {operation} function to geometry results."
    )

    return test_func


def __create_app_test(
    operation: str,
    param_creator: Callable[
        [], sip.BinningParam | sip.ResizeParam | sip.RotateParam
    ] = None,
) -> Callable[[], None]:
    """Create an app test function for a geometry transformation operation.

    Args:
        operation: Operation name (e.g., "rotate90", "fliph")
        param_creator: Function to create parameter object, if needed

    Returns:
        Test function
    """

    def test_func() -> None:
        """Test operation at application level through direct processing."""
        with datalab_test_app_context() as win:
            panel = win.imagepanel
            proc = panel.processor

            obj = __create_test_image()
            panel.add_object(obj)
            panel.objview.select_objects((1,))

            # Apply operation using app workflow
            param = param_creator() if param_creator else None
            if param is None:
                proc.run_feature(operation)
            else:
                proc.run_feature(operation, param=param)
            tr_obj = panel[len(panel)]

            # # Validate basic transformation requirements
            # __validate_basic_transformation(tr_obj, obj)

            # # Validate geometry transformation
            # tr_geometries = __get_geometries(tr_obj)
            # __validate_geometry_transformation(
            #     tr_geometries, or_geometries, operation, "App", param
            # )

        # Verify that image data is properly transformed
        __validate_image_data_transformation(tr_obj, obj, operation)

        # For exact image operations, validate ROI transformation
        if operation not in ["resize", "binning"]:
            __validate_roi_transformation(tr_obj, obj, operation)

    # Set proper function name and docstring
    test_func.__name__ = f"test_geometry_transform_{operation}_app"
    test_func.__doc__ = (
        f"Test {operation} operation at application level.\n\n"
        f"This test verifies the complete transformation workflow including "
        f"image processing and geometry transformation pipeline."
    )

    return test_func


# =============================================================================
# Results moved with the pixels
# =============================================================================


EXACT_OPERATIONS = ["rotate90", "rotate270", "fliph", "flipv", "transpose"]


def __create_feature_image(uniform: bool) -> sio.ImageObj:
    """Create an image with an off-center peak (pixel at row 11, column 17), on a
    non-square grid with an origin offset and anisotropic pixels (or non-uniform
    coordinates)."""
    nrows, ncols = 40, 64
    y, x = np.mgrid[0:nrows, 0:ncols]
    data = np.exp(-((x - 17.0) ** 2 / 20.0 + (y - 11.0) ** 2 / 8.0))
    obj = sio.create_image("Off-center feature", data)
    if uniform:
        obj.set_uniform_coords(0.5, 2.0, -3.0, 7.0)
    else:
        icol, irow = np.arange(ncols), np.arange(nrows)
        obj.set_coords(-3.0 + 0.5 * icol + 0.01 * icol**2, 7.0 + 2.0 * irow**1.2)
    return obj


def __peak_position(obj: sio.ImageObj) -> tuple[float, float]:
    """Return the physical coordinates of the center of the brightest pixel."""
    # pylint: disable-next=unbalanced-tuple-unpacking
    row, col = np.unravel_index(np.argmax(obj.data), obj.data.shape)
    if obj.is_uniform_coords:
        return obj.x0 + col * obj.dx, obj.y0 + row * obj.dy
    return obj.xcoords[col], obj.ycoords[row]


@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "non-uniform"])
@pytest.mark.parametrize("operation", EXACT_OPERATIONS)
def test_transformed_results_follow_the_pixels(operation: str, uniform: bool) -> None:
    """Analysis results moved by an image operation stay on the pixels they
    describe."""
    src = __create_feature_image(uniform)
    peak = sio.GeometryResult(
        "Peak", sio.KindShape.POINT, np.array([__peak_position(src)]), func_name="peak"
    )
    GeometryAdapter(peak).add_to(src)
    dst = GeometricTransformWrapper(getattr(sipi, operation), operation)(src)
    (moved,) = __get_geometries(dst)
    np.testing.assert_allclose(moved.coords[0], __peak_position(dst), atol=1e-12)


@pytest.mark.parametrize("operation", EXACT_OPERATIONS)
def test_transformed_results_match_recomputed_results(operation: str) -> None:
    """Analysis results moved by an image operation are the results of the same
    analysis on the transformed image."""
    src = __create_feature_image(uniform=True)
    GeometryAdapter(sipi.centroid(src)).add_to(src)
    dst = GeometricTransformWrapper(getattr(sipi, operation), operation)(src)
    (moved,) = __get_geometries(dst)
    recomputed = sipi.centroid(dst)
    # (the centroid is computed in a way which is sensitive to rounding noise)
    np.testing.assert_allclose(moved.coords, recomputed.coords, rtol=0, atol=1e-6)


# =============================================================================
# Parameter Creators
# =============================================================================


def __resize_param() -> sip.ResizeParam:
    """Create resize parameter."""
    return sip.ResizeParam.create(zoom=1.5)


def __binning_param() -> sip.BinningParam:
    """Create binning parameter."""
    return sip.BinningParam.create(sx=2, sy=3)


# =============================================================================
# Generate Test Functions
# =============================================================================

# Simple transformations (no parameters)
test_geometry_transform_rotate90_unit = __create_unit_test("rotate90")
test_geometry_transform_rotate90_app = __create_app_test("rotate90")

test_geometry_transform_rotate270_unit = __create_unit_test("rotate270")
test_geometry_transform_rotate270_app = __create_app_test("rotate270")

test_geometry_transform_fliph_unit = __create_unit_test("fliph")
test_geometry_transform_fliph_app = __create_app_test("fliph")

test_geometry_transform_flipv_unit = __create_unit_test("flipv")
test_geometry_transform_flipv_app = __create_app_test("flipv")

test_geometry_transform_transpose_unit = __create_unit_test("transpose")
test_geometry_transform_transpose_app = __create_app_test("transpose")

# Parametric transformations
test_geometry_transform_resize_unit = __create_unit_test("resize", __resize_param)
test_geometry_transform_resize_app = __create_app_test("resize", __resize_param)

test_geometry_transform_binning_unit = __create_unit_test("binning", __binning_param)
test_geometry_transform_binning_app = __create_app_test("binning", __binning_param)


if __name__ == "__main__":
    # test_geometry_transform_rotate90_unit()
    test_geometry_transform_rotate90_app()
    test_geometry_transform_rotate270_unit()
    test_geometry_transform_rotate270_app()
    test_geometry_transform_fliph_unit()
    test_geometry_transform_fliph_app()
    test_geometry_transform_flipv_unit()
    test_geometry_transform_flipv_app()
    test_geometry_transform_transpose_unit()
    test_geometry_transform_transpose_app()
    test_geometry_transform_resize_unit()
    test_geometry_transform_resize_app()
    test_geometry_transform_binning_unit()
    test_geometry_transform_binning_app()
    print("✅ All geometry transformation tests passed!")
