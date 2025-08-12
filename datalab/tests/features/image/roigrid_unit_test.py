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


if __name__ == "__main__":
    test_roi_grid_geometry_headless()
    test_roi_grid(screenshots=True)
