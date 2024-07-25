# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

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
from skimage import draw

import cdl.obj as dlo
import cdl.param as dlp
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.data import create_multigauss_image, create_paracetamol_signal

if TYPE_CHECKING:
    from cdl.core.gui.panel.image import ImagePanel
    from cdl.core.gui.panel.signal import SignalPanel


def __run_signal_computations(panel: SignalPanel, singleobj: bool | None = None):
    """Test all signal features related to ROI"""
    panel.processor.compute_fwhm(dlp.FWHMParam())
    panel.processor.compute_fw1e2()
    panel.processor.compute_histogram(dlp.HistogramParam())
    panel.remove_object()
    panel.processor.compute_roi_extraction(dlp.ROIDataParam.create(singleobj=singleobj))


SIZE = 200

# Image ROIs:
IROI1 = [SIZE // 2, SIZE // 2, SIZE - 25, SIZE]  # Rectangle
IROI2 = [SIZE // 4, SIZE // 2, SIZE // 2, SIZE // 2]  # Circle


def __run_image_computations(panel: ImagePanel, singleobj: bool | None = None):
    """Test all image features related to ROI"""
    panel.processor.compute_centroid()
    panel.processor.compute_enclosing_circle()
    panel.processor.compute_histogram(dlp.HistogramParam())
    panel.processor.compute_peak_detection(dlp.Peak2DDetectionParam())
    obj_nb = len(panel)
    panel.processor.compute_roi_extraction(dlp.ROIDataParam.create(singleobj=singleobj))

    if execenv.unattended:
        im0 = panel[obj_nb]
        if im0.roi is None:
            return

        # Assertions texts:
        nzroi = "Non-zero values expected in ROI"
        zroi = "Zero values expected outside ROI"
        roisham = "ROI shape mismatch"

        if singleobj is None or not singleobj:  # Multiple objects mode
            assert len(panel) == obj_nb + 2, "Two objects expected"
            im1, im2 = panel[obj_nb + 1], panel[obj_nb + 2]
            assert np.all(im1.data != 0), nzroi
            assert im1.data.shape == (IROI1[3] - IROI1[1], IROI1[2] - IROI1[0]), roisham
            assert np.all(im2.data != 0), nzroi
            assert im2.data.shape == (IROI2[2] - IROI2[0], IROI2[2] - IROI2[0]), roisham
            mask2 = np.zeros(shape=im2.data.shape, dtype=bool)
            xc, yc = (IROI2[0] + IROI2[2]) / 2, (IROI2[1] + IROI2[3]) / 2
            r = (IROI2[2] - IROI2[0]) / 2
            xc = yc = xc - IROI2[0]  # Adjust for ROI origin
            rr, cc = draw.disk((yc, xc), r)
            mask2[rr, cc] = 1
            assert np.all(im2.maskdata == ~mask2), "Mask data mismatch"
        else:  # Single object mode
            assert len(panel) == obj_nb + 1, "One object expected"

            # Compute ROI image masks:
            # (ROI are defined as [x1, y1, x2, y2] for rectangles
            # and [x1, y1, x2, y1] for circles:
            # x is the column index, y is the row index)
            mask1 = np.zeros(shape=(SIZE, SIZE), dtype=bool)
            mask1[IROI1[1] : IROI1[3], IROI1[0] : IROI1[2]] = 1
            xc, yc = (IROI2[0] + IROI2[2]) / 2, (IROI2[1] + IROI2[3]) / 2
            r = (IROI2[2] - IROI2[0]) / 2
            mask2 = np.zeros(shape=(SIZE, SIZE), dtype=bool)
            rr, cc = draw.disk((yc, xc), r)
            mask2[rr, cc] = 1
            mask = mask1 | mask2
            row_min = int(min(IROI1[1], IROI2[1] - r))
            col_min = int(min(IROI1[0], IROI2[0]))
            row_max = int(max(IROI1[3], IROI2[3] + r))
            col_max = int(max(IROI1[2], IROI2[2]))
            mask = mask[row_min:row_max, col_min:col_max]

            im = panel[obj_nb + 1]
            assert np.all(im.data[mask] != 0), nzroi
            assert np.all(im.data[~mask] == 0), zroi


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
    ima.data += 1  # Ensure that the image has non-zero values (for ROI check tests)
    ima.roi = np.array([IROI1, IROI2], int)
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


def test_roi_app(screenshots: bool = False):
    """Run ROI application test scenario"""
    with cdltest_app_context(console=False) as win:
        execenv.print("ROI application test:")
        # === Signal ROI extraction test ===
        panel = win.signalpanel
        sig1 = create_paracetamol_signal(SIZE)
        panel.add_object(sig1)
        __run_signal_computations(panel)
        sig2 = create_paracetamol_signal(SIZE)
        sig2.roi = np.array([[26, 41], [125, 146]], int)
        for singleobj in (False, True):
            sig2_i = sig2.copy()
            panel.add_object(sig2_i)
            print_obj_shapes(sig2_i)
            panel.processor.edit_regions_of_interest()
            if screenshots:
                win.statusBar().hide()
                win.take_screenshot("s_roi_signal")
            __run_signal_computations(panel, singleobj=singleobj)
        # === Image ROI extraction test ===
        panel = win.imagepanel
        param = dlo.new_image_param(height=SIZE, width=SIZE)
        ima1 = create_multigauss_image(param)
        panel.add_object(ima1)
        __run_image_computations(panel)
        ima2 = create_test_image_with_roi(param)
        for singleobj in (False, True):
            ima2_i = ima2.copy()
            panel.add_object(ima2_i)
            print_obj_shapes(ima2_i)
            panel.processor.edit_regions_of_interest()
            if screenshots:
                win.statusBar().hide()
                win.take_screenshot("i_roi_image")
            __run_image_computations(panel, singleobj=singleobj)


if __name__ == "__main__":
    test_roi_app(screenshots=True)
