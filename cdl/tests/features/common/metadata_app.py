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
# guitest: show

import numpy as np

import cdl.obj
import cdl.param
from cdl.core.gui.panel.base import BaseDataPanel
from cdl.core.gui.panel.image import ImagePanel
from cdl.core.gui.panel.signal import SignalPanel
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.data import create_paracetamol_signal
from cdl.tests.features.common import roi_app


def __run_signal_computations(panel: SignalPanel):
    """Test all signal features related to ROI"""
    execenv.print("  Signal features")
    panel.processor.compute_fwhm(cdl.param.FWHMParam())
    panel.processor.compute_fw1e2()


def __run_image_computations(panel: ImagePanel):
    """Test all image features related to ROI"""
    execenv.print("  Image features")
    panel.processor.compute_centroid()
    panel.processor.compute_enclosing_circle()
    panel.processor.compute_peak_detection(cdl.param.Peak2DDetectionParam())


def __test_metadata_features(panel: BaseDataPanel):
    """Test all metadata features"""
    # Duplicate the first object
    panel.duplicate_object()
    # Delete metadata of the first object
    panel.delete_metadata()
    # Select and copy metadata of the second object
    panel.objview.select_objects([2])
    panel.copy_metadata()
    # Select and paste metadata to the first object
    panel.objview.select_objects([1])
    panel.paste_metadata()


def test_metadata_app():
    """Run metadata application test scenario"""
    size = 200
    with cdltest_app_context() as win:
        execenv.print("Metadata application test:")
        # === Signal metadata features test ===
        panel = win.signalpanel
        sig = create_paracetamol_signal(size)
        sig.roi = np.array([[26, 41], [125, 146]], int)
        panel.add_object(sig)
        __run_signal_computations(panel)
        __test_metadata_features(panel)
        # === Image metadata features test ===
        panel = win.imagepanel
        param = cdl.obj.new_image_param(height=size, width=size)
        ima = roi_app.create_test_image_with_roi(param)
        panel.add_object(ima)
        __run_image_computations(panel)
        __test_metadata_features(panel)
        execenv.print("==> OK")


if __name__ == "__main__":
    test_metadata_app()
