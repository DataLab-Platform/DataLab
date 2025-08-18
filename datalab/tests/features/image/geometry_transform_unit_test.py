#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unit test for geometry transformation operations."""

from __future__ import annotations

import numpy as np
import sigima.objects
import sigima.proc.image as sigima_image
import sigima.proc.scalar
from sigima.tests import data as test_data

from datalab.adapters_metadata.geometry_adapter import GeometryAdapter
from datalab.config import Conf
from datalab.tests import datalab_test_app_context
from datalab.utils.geometry_transforms import apply_geometry_transform


def __create_test_image() -> sigima.objects.ImageObj:
    """Create a test image with geometry results for testing."""
    param = sigima.objects.Gauss2DParam.create(
        height=600,
        width=600,
        title="Test image (with geometry results)",
        dtype=sigima.objects.ImageDatatypes.UINT16,
    )
    obj = sigima.objects.create_image_from_param(param)
    for geometry in test_data.generate_geometry_results():
        GeometryAdapter(geometry).add_to(obj)
    obj.roi = sigima.objects.create_image_roi("rectangle", [10, 10, 50, 400])
    return obj


def __get_geometries(
    obj: sigima.objects.ImageObj,
) -> list[sigima.objects.GeometryResult]:
    """Get geometries from an image object."""
    return [ga.geometry for ga in GeometryAdapter.iterate_from_obj(obj)]


def __get_roi_coords(obj: sigima.objects.ImageObj) -> list[tuple]:
    """Get ROI coordinates from an image object."""
    return [sroi.get_physical_coords(obj) for sroi in obj.roi.single_rois]


def __validate_geometry_transformation(
    tr_geometries: list[sigima.objects.GeometryResult],
    or_geometries: list[sigima.objects.GeometryResult],
    test_context: str = "geometry",
) -> None:
    """Validate that geometry transformation was applied correctly.

    Args:
        tr_geometries: List of transformed geometry results
        or_geometries: List of original geometry results
        test_context: Context string for error messages (e.g., "App", "Unit")
    """
    for i, geometry in enumerate(tr_geometries):
        # Get the original geometry
        orig_geom = or_geometries[i]

        # Apply Sigima's rotate function directly for comparison
        expected_result = sigima.proc.scalar.rotate(orig_geom, np.pi / 2, None)

        # Compare the actual transformation result with expected
        np.testing.assert_allclose(
            geometry.coords,
            expected_result.coords,
            rtol=1e-10,
            err_msg=f"{test_context} geometry result {i} "
            f"({geometry.title}) coordinates do not match expected. "
            f"Got: {geometry.coords}, Expected: {expected_result.coords}",
        )

        # Verify other properties are preserved
        assert geometry.title == expected_result.title, (
            f"Title should be preserved for geometry {i}"
        )
        assert geometry.kind == expected_result.kind, (
            f"Kind should be preserved for geometry {i}"
        )
        np.testing.assert_array_equal(
            geometry.roi_indices,
            expected_result.roi_indices,
            err_msg=f"ROI indices should be preserved for geometry {i}",
        )


def __validate_basic_transformation(
    tr_obj: sigima.objects.ImageObj, or_obj: sigima.objects.ImageObj
) -> None:
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
    tr_obj: sigima.objects.ImageObj,
    or_obj: sigima.objects.ImageObj,
) -> None:
    """Validate that ROI is properly transformed.

    Args:
        tr_obj: The transformed image object
        or_obj: Original image object
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
        temp_geom = sigima.objects.GeometryResult(
            "temp_geom",
            sigima.objects.KindShape.RECTANGLE,
            coords=np.asarray([or_roi.get_physical_coords(or_obj)]),
        )
        exp_coords = sigima.proc.scalar.rotate(temp_geom, np.pi / 2, None).coords[0]
        np.testing.assert_array_equal(
            roi.get_physical_coords(tr_obj),
            exp_coords,
            err_msg=f"ROI {i} coordinates do not match expected.",
        )


def test_geometry_transform_rotate90_unit() -> None:
    """Test geometry transformation for 90-degree rotation at unit level.

    This test verifies that the DataLab geometry transformation properly
    applies Sigima's rotate function to geometry results. Sigima rotates
    geometries around their center point by default.
    """
    with datalab_test_app_context():
        obj = __create_test_image()

        # Store original data for comparison
        or_geometries = __get_geometries(obj)

        # Apply rotate90 transformation (same as DataLab processor does)
        # First apply the image transformation
        tr_obj = sigima_image.rotate90(obj)
        # Then apply the geometry transformation to the result
        apply_geometry_transform(tr_obj, "rotate90")

        # Validate basic transformation requirements
        __validate_basic_transformation(tr_obj, obj)

        # Validate geometry transformation
        tr_geometries = __get_geometries(tr_obj)
        __validate_geometry_transformation(tr_geometries, or_geometries, "Unit")

        # Validate ROI transformation
        __validate_roi_transformation(tr_obj, obj)


def test_geometry_transform_rotate90_app() -> None:
    """Test rotate90 operation at application level through the UI processor.

    This test verifies the complete application workflow including the
    image panel processor and geometry transformation pipeline.
    """
    with datalab_test_app_context() as win:
        with Conf.proc.keep_results.temp(True):
            obj = __create_test_image()

            # Store original data for comparison
            or_geometries = __get_geometries(obj)
            or_roi = __get_roi_coords(obj)

            # Apply rotate90 operation through the UI processor
            panel = win.imagepanel
            panel.add_object(obj)
            panel.objview.select_objects((1,))
            panel.processor.run_feature("rotate90")

            # Get the result object
            tr_obj = panel[len(panel)]

            # Validate basic transformation requirements
            __validate_basic_transformation(tr_obj, obj)

            # Validate geometry transformation
            tr_geometries = __get_geometries(tr_obj)
            __validate_geometry_transformation(tr_geometries, or_geometries, "App")

            # Verify that image data is properly rotated
            exp_data = np.rot90(obj.data)
            np.testing.assert_array_equal(
                tr_obj.data, exp_data, err_msg="Image data not properly rotated"
            )

            # Validate ROI transformation
            __validate_roi_transformation(tr_obj, obj)


if __name__ == "__main__":
    test_geometry_transform_rotate90_unit()
    test_geometry_transform_rotate90_app()
