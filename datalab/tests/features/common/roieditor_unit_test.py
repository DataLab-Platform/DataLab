# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI editor unit test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
from guidata.qthelpers import exec_dialog, qt_app_context
from sigima.objects import ImageROI, create_image_roi, create_signal_roi
from sigima.tests.data import create_multigaussian_image, create_paracetamol_signal

from datalab.env import execenv
from datalab.gui.roieditor import ImageROIEditor, SignalROIEditor
from datalab.utils import qthelpers as qth


def test_signal_roi_editor(screenshots: bool = False) -> None:
    """Test signal ROI editor"""
    cls = SignalROIEditor
    title = f"Testing {cls.__name__}"
    obj = create_paracetamol_signal()
    roi = create_signal_roi([36.4, 40.9], indices=False, title="Test ROI")
    obj.roi = roi
    with qt_app_context(exec_loop=False):
        execenv.print(title)
        roi_editor = cls(parent=None, obj=obj, mode="extract", size=(800, 600))
        if screenshots:
            roi_editor.show()
            qth.grab_save_window(roi_editor, "s_roi_editor")
        exec_dialog(roi_editor)


def create_image_roi_example() -> ImageROI:
    """Create an example image ROI"""
    roi = create_image_roi("rectangle", [720, 720, 304, 304], title="Test ROI 1")
    roi.add_roi(create_image_roi("circle", [550, 650, 165]), title="Test ROI 2")
    roi.add_roi(
        create_image_roi(
            "polygon", [225, 75, 650, 175, 625, 475, 200, 675], title="Test ROI 3"
        )
    )
    return roi


def test_image_roi_editor(screenshots: bool = False) -> None:
    """Test image ROI editor"""
    cls = ImageROIEditor
    title = f"Testing {cls.__name__}"
    obj = create_multigaussian_image()
    obj.roi = create_image_roi_example()
    with qt_app_context(exec_loop=False):
        execenv.print(title)
        for mode in ("extract", "apply"):
            execenv.print(f"  mode={mode}")
            roi_editor = cls(parent=None, obj=obj, mode=mode, size=(800, 600))
            if mode == "apply":
                # Clear the ROI
                roi_editor.remove_all_rois()
            if screenshots and mode == "extract":
                roi_editor.show()
                qth.grab_save_window(roi_editor, "i_roi_editor")
            if exec_dialog(roi_editor):
                results = roi_editor.get_roieditor_results()
                if results is not None:
                    edited_roi, modified = results
                    if mode == "extract":
                        # Test that the single ROIs are equal
                        # pylint: disable=use-a-generator
                        assert all(
                            [
                                np.array_equal(
                                    sroi1.get_physical_coords(obj),
                                    sroi2.get_physical_coords(obj),
                                )
                                for sroi1, sroi2 in zip(
                                    obj.roi.single_rois, edited_roi.single_rois
                                )
                            ]
                        ), "Single ROIs are not equal"
                        execenv.print("    Single ROIs indice coordinates:")
                        for sroi in edited_roi.single_rois:
                            execenv.print(
                                f"      {sroi.title} ({sroi.__class__.__name__}):"
                            )
                            c_i = [int(val) for val in sroi.get_indices_coords(obj)]
                            c_p = [float(val) for val in sroi.get_physical_coords(obj)]
                            execenv.print(f"        Indices : {c_i}")
                            execenv.print(f"        Physical: {c_p}")
                    else:
                        # Test the use case where the ROI is cleared
                        assert modified, "ROI is not modified"
                        assert edited_roi.is_empty(), "ROI is not cleared"


if __name__ == "__main__":
    test_signal_roi_editor(screenshots=True)
    test_image_roi_editor(screenshots=True)
