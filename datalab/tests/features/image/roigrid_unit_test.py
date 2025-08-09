# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""ROI grid unit test."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from guidata.qthelpers import exec_dialog, qt_app_context
from sigima.tests.data import create_grid_image

from datalab.gui.roigrideditor import ImageGridROIEditor
from datalab.utils import qthelpers as qth


def test_roi_grid(screenshots: bool = False) -> None:
    """ROI grid test."""
    with qt_app_context():
        roi_editor = ImageGridROIEditor(parent=None, obj=create_grid_image())
        if screenshots:
            roi_editor.show()
            qth.grab_save_window(roi_editor)
        exec_dialog(roi_editor)


if __name__ == "__main__":
    test_roi_grid(screenshots=True)
