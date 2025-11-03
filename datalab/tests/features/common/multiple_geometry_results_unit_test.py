# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test for multiple geometry results of the same kind
====================================================

This test verifies that multiple geometry results of the same kind (e.g., multiple
segment results like FWHM and FW1e2) can coexist on the same object.
"""

from __future__ import annotations

import numpy as np
import sigima.objects as sio
from sigima.objects.scalar import GeometryResult, KindShape

from datalab.adapters_metadata import GeometryAdapter


def test_multiple_segment_results():
    """Test that multiple segment geometry results can coexist."""
    # Create a simple signal object
    x = np.linspace(0, 10, 100)
    y = np.exp(-((x - 5) ** 2) / 2)
    obj = sio.create_signal("Test Signal", x, y)

    # Create two different segment results (simulating FWHM and FW1e2)
    fwhm_result = GeometryResult(
        title="fwhm",
        func_name="fwhm",
        kind=KindShape.SEGMENT,
        coords=np.array([[3.5, 0.6, 6.5, 0.6]]),  # x0, y0, x1, y1
        roi_indices=None,
        attrs={"method": "zero-crossing"},
    )

    fw1e2_result = GeometryResult(
        title="fw1e2",
        func_name="fw1e2",
        kind=KindShape.SEGMENT,
        coords=np.array([[3.0, 0.37, 7.0, 0.37]]),  # Different coordinates
        roi_indices=None,
        attrs={"method": "gaussian"},
    )

    # Add both results to the object
    GeometryAdapter(fwhm_result).add_to(obj)
    GeometryAdapter(fw1e2_result).add_to(obj)

    # Verify both results are stored
    results = list(GeometryAdapter.iterate_from_obj(obj))
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    # Verify we can retrieve both results by title
    titles = {adapter.title for adapter in results}
    assert "fwhm" in titles, "FWHM result not found"
    assert "fw1e2" in titles, "FW1e2 result not found"

    # Verify the results have different coordinates
    for adapter in results:
        if adapter.title == "fwhm":
            assert np.allclose(adapter.result.coords[0], [3.5, 0.6, 6.5, 0.6])
        elif adapter.title == "fw1e2":
            assert np.allclose(adapter.result.coords[0], [3.0, 0.37, 7.0, 0.37])

    print("✓ Multiple segment results can coexist")


def test_replace_same_title():
    """Test that adding a result with the same title replaces the old one."""
    # Create a simple signal object
    x = np.linspace(0, 10, 100)
    y = np.exp(-((x - 5) ** 2) / 2)
    obj = sio.create_signal("Test Signal", x, y)

    # Create first FWHM result
    fwhm_result_1 = GeometryResult(
        title="fwhm",
        kind=KindShape.SEGMENT,
        coords=np.array([[3.5, 0.6, 6.5, 0.6]]),
        roi_indices=None,
        func_name="fwhm",
        attrs={"method": "zero-crossing"},
    )

    # Create second FWHM result (same title, different coords)
    fwhm_result_2 = GeometryResult(
        title="fwhm",
        kind=KindShape.SEGMENT,
        coords=np.array([[3.6, 0.6, 6.6, 0.6]]),  # Slightly different
        roi_indices=None,
        func_name="fwhm",
        attrs={"method": "gaussian"},
    )

    # Add first result
    GeometryAdapter(fwhm_result_1).add_to(obj)

    # Verify one result
    results = list(GeometryAdapter.iterate_from_obj(obj))
    assert len(results) == 1, f"Expected 1 result after first add, got {len(results)}"

    # Add second result (should replace the first)
    GeometryAdapter(fwhm_result_2).add_to(obj)

    # Verify still only one result
    results = list(GeometryAdapter.iterate_from_obj(obj))
    assert len(results) == 1, f"Expected 1 result after second add, got {len(results)}"

    # Verify it's the second one (with updated coordinates)
    adapter = results[0]
    assert adapter.title == "fwhm"
    assert np.allclose(adapter.result.coords[0], [3.6, 0.6, 6.6, 0.6])
    assert adapter.result.attrs["method"] == "gaussian"

    print("✓ Results with same title are properly replaced")


if __name__ == "__main__":
    test_multiple_segment_results()
    test_replace_same_title()
    print("\n✅ All tests passed!")
