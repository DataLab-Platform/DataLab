# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ROI to plot item conversion unit tests
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
import sigima.objects
from guidata.qthelpers import qt_app_context
from sigima.tests.data import (
    create_multigaussian_image,
    create_paracetamol_signal,
    create_test_image_rois,
    create_test_signal_rois,
)

from datalab.adapters_plotpy import create_adapter_from_object, plotitem_to_singleroi
from datalab.env import execenv


def __conversion_methods(
    roi: sigima.objects.SignalROI | sigima.objects.ImageROI,
    obj: sigima.objects.SignalObj | sigima.objects.ImageObj,
) -> None:
    """Test conversion methods for single ROI objects"""
    execenv.print("    test `to_plot_item` and `from_plot_item` methods: ", end="")
    single_roi = roi.get_single_roi(0)
    with qt_app_context(exec_loop=False):
        plot_item = create_adapter_from_object(single_roi).to_plot_item(obj)
        sroi_new = plotitem_to_singleroi(plot_item, obj)
        orig_coords = [float(val) for val in single_roi.get_physical_coords(obj)]
        new_coords = [float(val) for val in sroi_new.get_physical_coords(obj)]
        execenv.print(f"{orig_coords} --> {new_coords}")
        # Check that coordinates are close
        # (allowing for rounding applied during conversion)
        assert np.allclose(orig_coords, new_coords, rtol=1e-5, atol=1e-10)


def test_signal_roi_plotitem_conversion() -> None:
    """Test signal ROIs conversion to/from plot items"""
    execenv.print("==============================================")
    execenv.print("Test signal ROIs conversion to/from plot items")
    execenv.print("==============================================")
    obj = create_paracetamol_signal()
    for roi in create_test_signal_rois(obj):
        __conversion_methods(roi, obj)


def test_image_roi_plotitem_conversion() -> None:
    """Test image ROIs conversion to/from plot items"""
    execenv.print("==============================================")
    execenv.print("Test image ROIs conversion to/from plot items")
    execenv.print("==============================================")
    obj = create_multigaussian_image()
    for roi in create_test_image_rois(obj):
        __conversion_methods(roi, obj)


if __name__ == "__main__":
    test_signal_roi_plotitem_conversion()
    test_image_roi_plotitem_conversion()
