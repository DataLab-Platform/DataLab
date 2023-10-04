# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
ROI test:

  - Defining Region of Interest on a signal
  - Defining Region of Interest on an image
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import cdl.obj as dlo
import cdl.param as dlp
from cdl.env import execenv
from cdl.tests import cdl_app_context
from cdl.tests.data import create_multigauss_image, create_paracetamol_signal

if TYPE_CHECKING:  # pragma: no cover
    from cdl.core.gui.panel.image import ImagePanel
    from cdl.core.gui.panel.signal import SignalPanel


def test_signal_features(panel: SignalPanel, singleobj: bool | None = None):
    """Test all signal features related to ROI"""
    panel.processor.compute_fwhm(dlp.FWHMParam())
    panel.processor.compute_fw1e2()
    panel.processor.compute_roi_extraction(dlp.ROIDataParam.create(singleobj=singleobj))


def test_image_features(panel: ImagePanel, singleobj: bool | None = None):
    """Test all image features related to ROI"""
    panel.processor.compute_centroid()
    panel.processor.compute_enclosing_circle()
    panel.processor.compute_peak_detection(dlp.Peak2DDetectionParam())
    panel.processor.compute_roi_extraction(dlp.ROIDataParam.create(singleobj=singleobj))


def create_test_image_with_roi(
    newimageparam: dlo.NewImageParam,
) -> dlo.ImageObj:
    """Create test image with ROIs

    Args:
        newimageparam (cdl.obj.NewImageParam): Image parameters

    Returns:
        cdl.obj.ImageObj: Image object with ROIs
    """
    ima = create_multigauss_image(newimageparam)
    dy, dx = ima.size
    roi1 = [dx // 2, dy // 2, dx - 25, dy]
    roi2 = [dx // 4, dy // 2, dx // 2, dy // 2]
    ima.roi = np.array([roi1, roi2], int)
    return ima


def array_2d_to_str(arr: np.ndarray) -> str:
    """Return 2-D array characteristics as string"""
    return f"{arr.shape[0]} x {arr.shape[1]} array (min={arr.min()}, max={arr.max()})"


def array_1d_to_str(arr: np.ndarray) -> str:
    """Return 1-D array characteristics as string"""
    return f"{arr.size} columns array (min={arr.min()}, max={arr.max()})"


def print_obj_shapes(obj):
    """Print object and associated ROI array shapes"""
    execenv.print(f"  Accessing object '{obj.title}':")
    func = array_1d_to_str if isinstance(obj, dlo.SignalObj) else array_2d_to_str
    execenv.print(f"    data: {func(obj.data)}")
    if obj.roi is not None:
        for idx in range(obj.roi.shape[0]):
            roi_data = obj.get_data(idx)
            if isinstance(obj, dlo.SignalObj):
                roi_data = roi_data[1]  # y data
            execenv.print(f"    ROI[{idx}]: {func(roi_data)}")


def test():
    """Run ROI application test scenario"""
    size = 200
    with cdl_app_context() as win:
        execenv.print("ROI application test:")
        # === Signal ROI extraction test ===
        panel = win.signalpanel
        sig1 = create_paracetamol_signal(size)
        panel.add_object(sig1)
        test_signal_features(panel)
        sig2 = create_paracetamol_signal(size)
        sig2.roi = np.array([[26, 41], [125, 146]], int)
        for singleobj in (False, True):
            sig2_i = sig2.copy()
            panel.add_object(sig2_i)
            print_obj_shapes(sig2_i)
            panel.processor.edit_regions_of_interest()
            win.take_screenshot("s_roi_signal")
            test_signal_features(panel, singleobj=singleobj)
        # === Image ROI extraction test ===
        panel = win.imagepanel
        param = dlo.new_image_param(height=size, width=size)
        ima1 = create_multigauss_image(param)
        panel.add_object(ima1)
        test_image_features(panel)
        ima2 = create_test_image_with_roi(param)
        for singleobj in (False, True):
            ima2_i = ima2.copy()
            panel.add_object(ima2_i)
            print_obj_shapes(ima2_i)
            panel.processor.edit_regions_of_interest()
            win.take_screenshot("i_roi_image")
            test_image_features(panel, singleobj=singleobj)


if __name__ == "__main__":
    test()
