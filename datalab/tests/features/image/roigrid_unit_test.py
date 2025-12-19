# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""ROI grid unit test."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from guidata.qthelpers import exec_dialog, qt_app_context
from sigima.proc.image import Direction, ROIGridParam
from sigima.tests.data import create_grid_of_gaussian_images

from datalab.gui.roigrideditor import ImageGridROIEditor
from datalab.utils import qthelpers as qth


def test_roi_grid(screenshots: bool = False) -> None:
    """ROI grid test."""
    with qt_app_context():
        roi_editor = ImageGridROIEditor(
            parent=None, obj=create_grid_of_gaussian_images()
        )
        if screenshots:
            roi_editor.show()
            qth.grab_save_window(roi_editor)
        exec_dialog(roi_editor)


def test_roi_grid_geometry_headless() -> None:
    """Test ROI grid geometry in headless mode."""
    img = create_grid_of_gaussian_images()

    # Create grid parameters
    gp = ROIGridParam()
    gp.nx, gp.ny = 2, 2
    gp.xsize = gp.ysize = 50
    gp.xtranslation = gp.ytranslation = 50
    gp.xdirection = gp.ydirection = Direction.INCREASING

    with qt_app_context():
        dlg = ImageGridROIEditor(parent=None, obj=img, gridparam=gp)
        # Set a small grid and sizes
        dlg.update_obj(update_item=False)
        roi = dlg.get_roi()
        assert roi is not None
        # 4 ROIs, centered in each cell
        assert len(list(roi)) == 4
        titles = [r.title for r in roi]
        assert "ROI(1,1)" in titles and "ROI(2,2)" in titles
        # Check one ROI position approximately
        r00 = next(r for r in roi if r.title == "ROI(1,1)")
        _x0, _y0, dx, dy = r00.get_physical_coords(img)
        assert dx == img.width / 2 * 0.5
        assert dy == img.height / 2 * 0.5


def test_roi_grid_custom_step() -> None:
    """Test ROI grid with custom xstep/ystep parameters.

    This test verifies the bug fix for Issue #XXX where grid ROI extraction
    was not working correctly for images with non-uniformly distributed features
    (e.g., laser spot arrays with gaps between spots).

    The bug was that xstep and ystep parameters were missing, so users couldn't
    adjust the spacing between ROIs when spots don't fill the entire image.
    """
    img = create_grid_of_gaussian_images()

    # Test Case 1: Custom step to simulate tighter spacing (e.g., laser spots)
    gp = ROIGridParam()
    gp.nx, gp.ny = 3, 3
    gp.xsize = gp.ysize = 20  # Small ROI size (20% of cell)
    gp.xtranslation = gp.ytranslation = 50  # Centered
    gp.xstep = gp.ystep = 75  # Tighter spacing (75% instead of 100%)
    gp.xdirection = gp.ydirection = Direction.INCREASING

    with qt_app_context():
        dlg = ImageGridROIEditor(parent=None, obj=img, gridparam=gp)
        dlg.update_obj(update_item=False)
        roi = dlg.get_roi()
        assert roi is not None
        # 9 ROIs for 3x3 grid
        assert len(list(roi)) == 9

        # Verify spacing is correctly applied
        r11 = next(r for r in roi if r.title == "ROI(1,1)")
        r12 = next(r for r in roi if r.title == "ROI(1,2)")
        x0_r11, _, _, _ = r11.get_physical_coords(img)
        x0_r12, _, _, _ = r12.get_physical_coords(img)

        # Expected spacing: (width / nx) * (xstep / 100)
        expected_spacing = (img.width / gp.nx) * (gp.xstep / 100.0)
        actual_spacing = x0_r12 - x0_r11

        # Allow 1% tolerance for numerical precision
        assert abs(actual_spacing - expected_spacing) / expected_spacing < 0.01

    # Test Case 2: Different X and Y steps
    gp2 = ROIGridParam()
    gp2.nx, gp2.ny = 2, 2
    gp2.xsize = gp2.ysize = 30
    gp2.xtranslation = gp2.ytranslation = 50
    gp2.xstep = 80  # Tighter X spacing
    gp2.ystep = 120  # Wider Y spacing
    gp2.xdirection = gp2.ydirection = Direction.INCREASING

    with qt_app_context():
        dlg2 = ImageGridROIEditor(parent=None, obj=img, gridparam=gp2)
        dlg2.update_obj(update_item=False)
        roi2 = dlg2.get_roi()
        assert roi2 is not None
        assert len(list(roi2)) == 4

        # Verify X spacing (80%)
        r11 = next(r for r in roi2 if r.title == "ROI(1,1)")
        r12 = next(r for r in roi2 if r.title == "ROI(1,2)")
        x0_r11, y0_r11, _, _ = r11.get_physical_coords(img)
        x0_r12, _, _, _ = r12.get_physical_coords(img)
        expected_x_spacing = (img.width / gp2.nx) * 0.8
        actual_x_spacing = x0_r12 - x0_r11
        assert abs(actual_x_spacing - expected_x_spacing) / expected_x_spacing < 0.01

        # Verify Y spacing (120%)
        r21 = next(r for r in roi2 if r.title == "ROI(2,1)")
        _, y0_r21, _, _ = r21.get_physical_coords(img)
        expected_y_spacing = (img.height / gp2.ny) * 1.2
        actual_y_spacing = y0_r21 - y0_r11
        assert abs(actual_y_spacing - expected_y_spacing) / expected_y_spacing < 0.01


if __name__ == "__main__":
    test_roi_grid_geometry_headless()
    test_roi_grid_custom_step()
    test_roi_grid(screenshots=True)
