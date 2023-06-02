# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Metadata application test:

  - Create signal/image, with ROI
  - Compute things (adds metadata)
  - Test metadata delete, copy, paste
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np

import cdl.param
from cdl.core.gui.panel.base import BaseDataPanel
from cdl.core.gui.panel.image import ImagePanel
from cdl.core.gui.panel.signal import SignalPanel
from cdl.env import execenv
from cdl.tests import cdl_app_context
from cdl.tests.data import create_test_signal1
from cdl.tests.features.common import roi_app

SHOW = True  # Show test in GUI-based test launcher


def test_signal_features(panel: SignalPanel):
    """Test all signal features related to ROI"""
    panel.processor.compute_fwhm(cdl.param.FWHMParam())
    panel.processor.compute_fw1e2()


def test_image_features(panel: ImagePanel):
    """Test all image features related to ROI"""
    panel.processor.compute_centroid()
    panel.processor.compute_enclosing_circle()
    panel.processor.compute_peak_detection(cdl.param.Peak2DDetectionParam())


def test_metadata_features(panel: BaseDataPanel):
    """Test all metadata features"""
    panel.duplicate_object()
    panel.delete_metadata()
    panel.objview.select_nums([-2])
    panel.copy_metadata()
    panel.objview.select_nums([-1])
    panel.paste_metadata()


def test():
    """Run ROI unit test scenario"""
    size = 200
    with cdl_app_context() as win:
        execenv.print("Metadata application test:")
        # === Signal metadata features test ===
        panel = win.signalpanel
        sig = create_test_signal1(size)
        sig.roi = np.array([[26, 41], [125, 146]], int)
        panel.add_object(sig)
        test_signal_features(panel)
        test_metadata_features(panel)
        # === Image metadata features test ===
        panel = win.imagepanel
        ima = roi_app.create_test_image_with_roi(size)
        panel.add_object(ima)
        test_image_features(panel)
        test_metadata_features(panel)


if __name__ == "__main__":
    test()
