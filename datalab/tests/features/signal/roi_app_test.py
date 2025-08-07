# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal ROI application test"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import sigima.params as sigima_param
from sigima.objects import SignalROI, create_signal_roi
from sigima.tests.data import create_paracetamol_signal
from sigima.tests.helpers import print_obj_data_dimensions

from datalab.env import execenv
from datalab.tests import datalab_test_app_context

if TYPE_CHECKING:
    from datalab.gui.panel.signal import SignalPanel

SIZE = 200

# Signal ROIs:
SROI1 = [26, 41]
SROI2 = [125, 146]


def __run_signal_computations(panel: SignalPanel, singleobj: bool | None = None):
    """Test all signal features related to ROI"""
    panel.processor.run_feature("fwhm", sigima_param.FWHMParam())
    panel.processor.run_feature("fw1e2")
    panel.processor.run_feature("histogram", sigima_param.HistogramParam())
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

    panel.processor.run_feature(
        "gaussian_filter", sigima_param.GaussianParam.create(sigma=10.0)
    )
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


def test_signal_roi_app(screenshots: bool = False) -> None:
    """Run Signal ROI application test scenario

    Args:
        screenshots: If True, take screenshots during the test.
    """
    with datalab_test_app_context(console=False) as win:
        execenv.print("Signal ROI application test:")
        panel = win.signalpanel
        sig1 = create_paracetamol_signal(SIZE)
        panel.add_object(sig1)
        __run_signal_computations(panel)
        sig2 = create_paracetamol_signal(SIZE)
        sig2.roi = create_signal_roi([SROI1, SROI2], indices=True)
        for singleobj in (False, True):
            sig2_i = sig2.copy()
            panel.add_object(sig2_i)
            print_obj_data_dimensions(sig2_i, indent=1)
            panel.processor.edit_roi_graphically()
            if screenshots:
                win.statusBar().hide()
                win.take_screenshot("s_roi_signal")
            __run_signal_computations(panel, singleobj=singleobj)


@pytest.mark.skip(reason="This test is only for manual testing")
def test_signal_roi_basic_app():
    """Run Signal ROI basic application test scenario"""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        sig1 = create_paracetamol_signal(SIZE)
        panel.add_object(sig1)
        panel.processor.edit_roi_graphically()


if __name__ == "__main__":
    test_signal_roi_basic_app()
    test_signal_roi_app(screenshots=False)
