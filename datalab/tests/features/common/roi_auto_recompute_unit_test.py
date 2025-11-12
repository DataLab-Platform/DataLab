# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit test for automatic recomputation of 1-to-0 analysis operations after ROI changes.

This test verifies that analysis results (like centroid) are automatically updated
when ROI is modified through various methods:
- Programmatically setting a new ROI (simulating edit_roi_graphically)
- Adding multiple ROIs at once
- Deleting a single ROI using delete_single_roi
- Deleting all ROIs using delete_regions_of_interest

The test creates a Gaussian image, computes its centroid, then verifies that:
1. The centroid changes when a ROI is added to restrict the calculation region
2. Two centroid rows are generated when two ROIs are added
3. One centroid row remains after deleting the first ROI
4. The centroid returns to the original value when all ROIs are deleted
"""

# guitest: show

from __future__ import annotations

import numpy as np
from sigima.objects import Gauss2DParam, create_image_from_param, create_image_roi

from datalab.adapters_metadata import GeometryAdapter
from datalab.config import Conf
from datalab.tests import datalab_test_app_context


def get_centroid_coords(obj) -> tuple[float, float] | None:
    """Extract centroid coordinates from object metadata.

    Args:
        obj: Image object with centroid results

    Returns:
        Tuple (x, y) of centroid coordinates, or None if no centroid found
    """
    adapter = GeometryAdapter.from_obj(obj, "centroid")
    if adapter is not None:
        print("adapter.result.coords:", adapter.result.coords)
        return tuple(adapter.result.coords[0])
    return None


def test_roi_auto_recompute():
    """Test automatic recomputation of analysis results when ROI changes."""
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel

        # Create a 2D Gaussian image in a 200x200 image
        SIZE = 200
        param = Gauss2DParam.create(height=SIZE, width=SIZE, sigma=20)
        img = create_image_from_param(param)
        panel.add_object(img)

        # Compute centroid on the full image
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("centroid")
        centroid = get_centroid_coords(img)
        assert centroid is not None, "Centroid should be computed"
        x0, y0 = centroid
        print(f"\nInitial centroid (full image): ({x0:.1f}, {y0:.1f})")

        # Step 1: Add ROI (simulating edit_roi_graphically)
        # Add a rectangular ROI in the upper-left quadrant
        roi = create_image_roi("rectangle", [25, 25, 50, 50])  # x0, y0, width, height
        img.roi = roi
        panel.refresh_plot("selected", update_items=True)
        # Trigger auto-recompute by simulating ROI modification
        panel.processor.auto_recompute_analysis(img)

        # Verify centroid was updated
        centroid = get_centroid_coords(img)
        assert centroid is not None, "Centroid should still exist after ROI change"
        x1, y1 = centroid
        print(f"Centroid after adding ROI (upper-left): ({x1:.1f}, {y1:.1f})")
        # Centroid should have moved (different from initial)
        centroid_changed = not (
            np.isclose(x1, x0, atol=0.1) and np.isclose(y1, y0, atol=0.1)
        )
        assert centroid_changed, (
            f"Centroid should have changed from ({x0:.1f}, {y0:.1f}) "
            f"to ({x1:.1f}, {y1:.1f})"
        )

        # Step 2: Add two ROIs at once
        # Create a multi-ROI object with two rectangular regions
        roi1 = create_image_roi("rectangle", [25, 25, 50, 50])  # Upper-left
        roi2 = create_image_roi("rectangle", [150, 150, 40, 40])  # Lower-right
        roi1.add_roi(roi2)  # Combine both ROIs
        img.roi = roi1
        panel.refresh_plot("selected", update_items=True)
        panel.processor.auto_recompute_analysis(img)

        # Verify centroid now has TWO rows (one for each ROI)
        adapter = GeometryAdapter.from_obj(img, "centroid")
        assert adapter is not None, "Centroid adapter should exist"
        coords = adapter.result.coords
        print(f"Centroid coords after adding two ROIs:\n{coords}")
        assert len(coords) == 2, f"Should have 2 centroid rows, got {len(coords)}"
        x2_roi0, y2_roi0 = coords[0]
        x2_roi1, y2_roi1 = coords[1]
        print(
            f"Centroid ROI 0 (upper-left): ({x2_roi0:.1f}, {y2_roi0:.1f}), "
            f"ROI 1 (lower-right): ({x2_roi1:.1f}, {y2_roi1:.1f})"
        )

        # Step 3: Delete the first ROI using delete_single_roi
        panel.processor.delete_single_roi(roi_index=0)

        # Verify centroid now has ONE row (for the remaining ROI)
        adapter = GeometryAdapter.from_obj(img, "centroid")
        assert adapter is not None, "Centroid adapter should exist after ROI deletion"
        coords = adapter.result.coords
        print(f"Centroid coords after deleting first ROI:\n{coords}")
        assert len(coords) == 1, f"Should have 1 centroid row, got {len(coords)}"
        x3, y3 = coords[0]
        print(f"Centroid after deleting first ROI: ({x3:.1f}, {y3:.1f})")
        # The remaining ROI should be the second one (lower-right), so centroid should
        # be close to what was previously in ROI 1
        assert np.isclose(x3, x2_roi1, atol=1.0), (
            f"X centroid should be close to {x2_roi1:.1f}, got {x3:.1f}"
        )
        assert np.isclose(y3, y2_roi1, atol=1.0), (
            f"Y centroid should be close to {y2_roi1:.1f}, got {y3:.1f}"
        )

        # Step 4: Delete all remaining ROIs using delete_regions_of_interest
        panel.processor.delete_regions_of_interest()

        # Verify centroid was updated back to original
        centroid = get_centroid_coords(img)
        assert centroid is not None, "Centroid should exist after deleting all ROIs"
        x4, y4 = centroid
        print(f"Centroid after delete_regions_of_interest: ({x4:.1f}, {y4:.1f})")
        # Should return to original centroid of full image
        assert np.isclose(x4, x0, atol=1.0), (
            f"X centroid should return to {x0:.1f}, got {x4:.1f}"
        )
        assert np.isclose(y4, y0, atol=1.0), (
            f"Y centroid should return to {y0:.1f}, got {y4:.1f}"
        )

        print("\n✓ All ROI auto-recompute tests passed!")


if __name__ == "__main__":
    test_roi_auto_recompute()
