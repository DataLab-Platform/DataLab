# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI editor unit test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
from guidata.qthelpers import exec_dialog, qt_app_context
from plotpy.plot import PlotDialog
from sigima.obj import ImageROI, create_image_roi, create_signal_roi
from sigima.tests.data import create_multigauss_image, create_paracetamol_signal

from datalab.env import execenv
from datalab.gui.panel.image import ImagePanel
from datalab.gui.panel.signal import SignalPanel
from datalab.gui.roieditor import ImageROIEditor, SignalROIEditor


def test_signal_roi_editor() -> None:
    """Test signal ROI editor"""
    cls = SignalROIEditor
    title = f"Testing {cls.__name__}"
    options = SignalPanel.ROIDIALOGOPTIONS
    obj = create_paracetamol_signal()
    roi = create_signal_roi([50, 100], indices=True)
    obj.roi = roi
    with qt_app_context(exec_loop=False):
        execenv.print(title)
        dlg = PlotDialog(title=title, edit=True, options=options, toolbar=True)
        editor = cls(dlg, obj, extract=True)
        dlg.button_layout.insertWidget(0, editor)
        exec_dialog(dlg)


def create_image_roi_example() -> ImageROI:
    """Create an example image ROI"""
    roi = create_image_roi("rectangle", [500, 750, 1000, 1250])
    roi.add_roi(create_image_roi("circle", [1500, 1500, 500]))
    roi.add_roi(
        create_image_roi("polygon", [450, 150, 1300, 350, 1250, 950, 400, 1350])
    )
    return roi


def test_image_roi_editor() -> None:
    """Test image ROI editor"""
    cls = ImageROIEditor
    title = f"Testing {cls.__name__}"
    options = ImagePanel.ROIDIALOGOPTIONS
    obj = create_multigauss_image()
    obj.roi = create_image_roi_example()
    with qt_app_context(exec_loop=False):
        execenv.print(title)
        for extract in (True, False):
            execenv.print(f"  extract={extract}")
            dlg = PlotDialog(title=title, edit=True, options=options, toolbar=True)
            roi_editor = cls(dlg, obj, extract=extract)
            dlg.button_layout.insertWidget(0, roi_editor)
            if not extract:
                # Clear the ROI
                roi_editor.remove_all_rois()
            if exec_dialog(dlg):
                results = roi_editor.get_roieditor_results()
                if results is not None:
                    edited_roi, modified = results
                    if extract:
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
    test_signal_roi_editor()
    test_image_roi_editor()
