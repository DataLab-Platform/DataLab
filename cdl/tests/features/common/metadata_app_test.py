# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Metadata application test:

  - Create signal/image, with ROI
  - Compute things (adds metadata)
  - Test metadata delete, copy, paste
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdl.computation.image as cpi
import cdl.computation.image.detection as cpi_det
import cdl.computation.signal as cps
import cdl.obj as dlo
import cdl.param as dlp
from cdl.core.gui.panel.base import BaseDataPanel
from cdl.core.gui.panel.image import ImagePanel
from cdl.core.gui.panel.signal import SignalPanel
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.data import create_paracetamol_signal
from cdl.tests.features.common import roi_app_test


def __run_signal_computations(panel: SignalPanel):
    """Test all signal features related to ROI"""
    execenv.print("  Signal features")
    panel.processor.run_feature(cps.fwhm, dlp.FWHMParam())
    panel.processor.run_feature(cps.fw1e2)


def __run_image_computations(panel: ImagePanel):
    """Test all image features related to ROI"""
    execenv.print("  Image features")
    panel.processor.run_feature(cpi.centroid)
    panel.processor.run_feature(cpi.enclosing_circle)
    panel.processor.run_feature(cpi_det.peak_detection, dlp.Peak2DDetectionParam())


def __test_metadata_features(panel: BaseDataPanel):
    """Test all metadata features"""
    # Duplicate the first object
    panel.duplicate_object()
    # Delete metadata of the first object
    for keep_roi in (True, False):  # Test both cases (coverage test)
        panel.delete_metadata(keep_roi=keep_roi)
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
        sig.roi = dlo.create_signal_roi([[26, 41], [125, 146]], indices=True)
        panel.add_object(sig)
        __run_signal_computations(panel)
        __test_metadata_features(panel)
        # === Image metadata features test ===
        panel = win.imagepanel
        param = dlo.new_image_param(height=size, width=size)
        ima = roi_app_test.create_test_image_with_roi(param)
        panel.add_object(ima)
        __run_image_computations(panel)
        __test_metadata_features(panel)
        execenv.print("==> OK")


if __name__ == "__main__":
    test_metadata_app()
