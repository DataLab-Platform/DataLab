# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal ROI application test"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sigima.params as sigima_param
from sigima.objects import SignalROI, create_signal_roi
from sigima.tests.data import create_paracetamol_signal
from sigima.tests.helpers import print_obj_data_dimensions

from datalab.config import Conf
from datalab.env import execenv
from datalab.tests import datalab_test_app_context

if TYPE_CHECKING:
    from datalab.gui.panel.signal import SignalPanel

SIZE = 200

# Signal ROIs:
SROI1 = [26, 41]
SROI2 = [125, 146]


def __run_signal_computations(panel: SignalPanel):
    """Test all signal features related to ROI"""
    panel.processor.run_feature("fwhm", sigima_param.FWHMParam())
    panel.processor.run_feature("fw1e2")
    panel.processor.run_feature("histogram", sigima_param.HistogramParam())
    roi = SignalROI()
    panel.processor.compute_roi_extraction(roi)


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
            with Conf.proc.extract_roi_singleobj.temp(singleobj):
                sig2_i = sig2.copy()
                panel.add_object(sig2_i)
                print_obj_data_dimensions(sig2_i, indent=1)
                panel.processor.edit_roi_graphically()
                if screenshots:
                    win.statusBar().hide()
                    win.take_screenshot("s_roi_signal")
                __run_signal_computations(panel)


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
