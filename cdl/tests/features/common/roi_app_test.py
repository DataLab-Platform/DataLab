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

import sigima_.param as sp
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.data import create_multigauss_image, create_paracetamol_signal
from sigima_ import (
    ImageObj,
    ImageROI,
    NewImageParam,
    SignalObj,
    SignalROI,
    create_image_roi,
    create_signal_roi,
)

if TYPE_CHECKING:
    from cdl.gui.panel.image import ImagePanel
    from cdl.gui.panel.signal import SignalPanel

SIZE = 200

# Signal ROIs:
SROI1 = [26, 41]
SROI2 = [125, 146]

# Image ROIs:
IROI1 = [SIZE // 2, SIZE // 2, SIZE - 25 - SIZE // 2, SIZE - SIZE // 2]  # Rectangle
IROI2 = [SIZE // 3, SIZE // 2, SIZE // 4]  # Circle
IROI3 = [
    SIZE // 2,
    SIZE // 2,
    SIZE // 2,
    SIZE - SIZE // 4,
    SIZE - SIZE // 4,
    SIZE - SIZE // 3,
]  # Polygon (triangle, that is intentionally inside the rectangle, so that this ROI
# has no impact on the mask calculations in the tests)


def __run_signal_computations(panel: SignalPanel, singleobj: bool | None = None):
    """Test all signal features related to ROI"""
    panel.processor.run_feature("fwhm", sp.FWHMParam())
    panel.processor.run_feature("fw1e2")
    panel.processor.run_feature("histogram", sp.HistogramParam())
    panel.remove_object()
    obj_nb = len(panel)
    last_obj = panel[obj_nb]
    roi = SignalROI(singleobj=singleobj)
    if execenv.unattended:
        # In unattended mode, we need to set the ROI manually.
        # On the contrary, in interactive mode, the ROI editor is opened and will
        # automatically set the ROI from the currently selected object.
        if last_obj.roi is not None:
            roi.single_rois = last_obj.roi.single_rois

    panel.processor.run_feature("gaussian_filter", sp.GaussianParam.create(sigma=10.0))
    if execenv.unattended and last_obj.roi is not None and not last_obj.roi.is_empty():
        # Check if the processed data is correct: signal should be the same as the
        # original data outside the ROI, and should be different inside the ROI.
        orig = last_obj.data
        new = panel[obj_nb + 1].data
        assert not np.any(new[SROI1[0] : SROI1[1]] == orig[SROI1[0] : SROI1[1]]), (
            "Signal ROI 1 data mismatch"
        )
        assert not np.any(new[SROI2[0] : SROI2[1]] == orig[SROI2[0] : SROI2[1]]), (
            "Signal ROI 2 data mismatch"
        )
        assert np.all(new[: SROI1[0]] == orig[: SROI1[0]]), (
            "Signal before ROI 1 data mismatch"
        )
        assert np.all(new[SROI1[1] : SROI2[0]] == orig[SROI1[1] : SROI2[0]]), (
            "Signal between ROIs data mismatch"
        )
        assert np.all(new[SROI2[1] :] == orig[SROI2[1] :]), (
            "Signal after ROI 2 data mismatch"
        )
    panel.remove_object()

    panel.processor.compute_roi_extraction(roi)
    if execenv.unattended and last_obj.roi is not None and not last_obj.roi.is_empty():
        # Assertions texts:
        ssm = "Signal %d size mismatch"
        sdm = "Signal %d data mismatch"

        orig = last_obj.data
        if singleobj is None or not singleobj:  # Multiple objects mode
            assert len(panel) == obj_nb + 2, "Two objects expected"
            sig1, sig2 = panel[obj_nb + 1], panel[obj_nb + 2]
            assert sig1.data.size == SROI1[1] - SROI1[0], ssm % 1
            assert sig2.data.size == SROI2[1] - SROI2[0], ssm % 2
            assert np.all(sig1.data == orig[SROI1[0] : SROI1[1]]), sdm % 1
            assert np.all(sig2.data == orig[SROI2[0] : SROI2[1]]), sdm % 2
        else:
            assert len(panel) == obj_nb + 1, "One object expected"
            sig = panel[obj_nb + 1]
            exp_size = SROI1[1] - SROI1[0] + SROI2[1] - SROI2[0]
            assert sig.data.size == exp_size, "Signal size mismatch"
            assert np.all(
                sig.data[: SROI1[1] - SROI1[0]] == orig[SROI1[0] : SROI1[1]]
            ), sdm % 1
            assert np.all(
                sig.data[SROI2[0] - SROI2[1] :] == orig[SROI2[0] : SROI2[1]]
            ), sdm % 2


def __run_image_computations(panel: ImagePanel, singleobj: bool | None = None):
    """Test all image features related to ROI"""
    panel.processor.run_feature("centroid")
    panel.processor.run_feature("enclosing_circle")
    panel.processor.run_feature("histogram", sp.HistogramParam())
    panel.processor.run_feature("peak_detection", sp.Peak2DDetectionParam())
    obj_nb = len(panel)
    last_obj = panel[obj_nb]
    roi = ImageROI(singleobj=singleobj)
    if execenv.unattended:
        # In unattended mode, we need to set the ROI manually.
        # On the contrary, in interactive mode, the ROI editor is opened and will
        # automatically set the ROI from the currently selected object.
        if last_obj.roi is not None:
            roi.single_rois = last_obj.roi.single_rois

    value = 1
    panel.processor.run_feature(
        "addition_constant", sp.ConstantParam.create(value=value)
    )
    if execenv.unattended and last_obj.roi is not None and not last_obj.roi.is_empty():
        # Check if the processed data is correct: image should be the same as the
        # original data outside the ROI, and should be different inside the ROI.
        orig = last_obj.data
        new = panel[obj_nb + 1].data
        assert np.all(
            new[IROI1[1] : IROI1[3] + IROI1[1], IROI1[0] : IROI1[2] + IROI1[0]]
            == orig[IROI1[1] : IROI1[3] + IROI1[1], IROI1[0] : IROI1[2] + IROI1[0]]
            + value
        ), "Image ROI 1 data mismatch"
        assert np.all(
            new[IROI2[1] : IROI1[1] + 1, IROI2[0] : IROI2[0] + 2 * IROI2[2]]
            == orig[IROI2[1] : IROI1[1] + 1, IROI2[0] : IROI2[0] + 2 * IROI2[2]] + value
        ), "Image ROI 2 data mismatch"
        first_col = min(IROI1[0], IROI2[0] - IROI2[2])
        first_row = min(IROI1[1], IROI2[1] - IROI2[2])
        last_col = max(IROI1[0] + IROI1[2], IROI2[0] + 2 * IROI2[2])
        last_row = max(IROI1[1] + IROI1[3], IROI2[1] + 2 * IROI2[2])
        assert np.all(
            new[:first_row, :first_col] == np.array(orig[:first_row, :first_col], float)
        ), "Image before ROIs data mismatch"
        assert np.all(new[:first_row, last_col:] == orig[:first_row, last_col:]), (
            "Image after ROIs data mismatch"
        )
        assert np.all(new[last_row:, :first_col] == orig[last_row:, :first_col]), (
            "Image before ROIs data mismatch"
        )
        assert np.all(new[last_row:, last_col:] == orig[last_row:, last_col:]), (
            "Image after ROIs data mismatch"
        )
    panel.remove_object()

    panel.processor.compute_roi_extraction(roi)
    if execenv.unattended and last_obj.roi is not None and not last_obj.roi.is_empty():
        # Assertions texts:
        nzroi = "Non-zero values expected in ROI"
        zroi = "Zero values expected outside ROI"
        roisham = "ROI shape mismatch"

        if singleobj is None or not singleobj:  # Multiple objects mode
            assert len(panel) == obj_nb + 3, "Three objects expected"
            im1, im2 = panel[obj_nb + 1], panel[obj_nb + 2]
            assert np.all(im1.data != 0), nzroi
            assert im1.data.shape == (IROI1[3], IROI1[2]), roisham
            assert np.all(im2.data != 0), nzroi
            assert im2.data.shape == (IROI2[2] * 2, IROI2[2] * 2), roisham
            mask2 = np.zeros(shape=im2.data.shape, dtype=bool)
            xc, yc, r = IROI2
            xc = yc = r  # Adjust for ROI origin
            rr, cc = draw.disk((yc, xc), r)
            mask2[rr, cc] = 1
            assert np.all(im2.maskdata == ~mask2), "Mask data mismatch"
        else:  # Single object mode
            assert len(panel) == obj_nb + 1, "One object expected"

            mask1 = np.zeros(shape=(SIZE, SIZE), dtype=bool)
            mask1[IROI1[1] : IROI1[1] + IROI1[3], IROI1[0] : IROI1[0] + IROI1[2]] = 1
            xc, yc, r = IROI2
            mask2 = np.zeros(shape=(SIZE, SIZE), dtype=bool)
            rr, cc = draw.disk((yc, xc), r)
            mask2[rr, cc] = 1
            mask = mask1 | mask2
            row_min = int(min(IROI1[1], IROI2[1] - r))
            col_min = int(min(IROI1[0], IROI2[0] - r))
            row_max = int(max(IROI1[1] + IROI1[3], IROI2[1] + r))
            col_max = int(max(IROI1[0] + IROI1[2], IROI2[0] + r))
            mask = mask[row_min:row_max, col_min:col_max]

            # execenv.unattended = False
            im = panel[obj_nb + 1]
            assert np.all(im.data[mask] != 0), nzroi
            assert np.all(im.data[~mask] == 0), zroi


def create_test_image_with_roi(newimageparam: NewImageParam) -> ImageObj:
    """Create test image with ROIs

    Args:
        newimageparam (sigima_.NewImageParam): Image parameters

    Returns:
        sigima_.ImageObj: Image object with ROIs
    """
    ima = create_multigauss_image(newimageparam)
    ima.data += 1  # Ensure that the image has non-zero values (for ROI check tests)
    roi = create_image_roi("rectangle", IROI1)
    roi.add_roi(create_image_roi("circle", IROI2))
    roi.add_roi(create_image_roi("polygon", IROI3))
    ima.roi = roi
    return ima


def array_2d_to_str(arr: np.ndarray) -> str:
    """Return 2-D array characteristics as string"""
    if arr.size == 0:
        return "Empty array!"
    return f"{arr.shape[0]} x {arr.shape[1]} array (min={arr.min()}, max={arr.max()})"


def array_1d_to_str(arr: np.ndarray) -> str:
    """Return 1-D array characteristics as string"""
    if arr.size == 0:
        return "Empty array!"
    return f"{arr.size} columns array (min={arr.min()}, max={arr.max()})"


def print_obj_shapes(obj):
    """Print object and associated ROI array shapes"""
    execenv.print(f"  Accessing object '{obj.title}':")
    func = array_1d_to_str if isinstance(obj, SignalObj) else array_2d_to_str
    execenv.print(f"    data: {func(obj.data)}")
    if obj.roi is not None:
        for idx in range(len(obj.roi)):
            roi_data = obj.get_data(idx)
            if isinstance(obj, SignalObj):
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
        sig2.roi = create_signal_roi([SROI1, SROI2], indices=True)
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
        param = NewImageParam.create(height=SIZE, width=SIZE)
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
