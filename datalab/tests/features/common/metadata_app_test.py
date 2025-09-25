# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Metadata application test:

  - Create signal/image, with ROI
  - Compute things (adds metadata)
  - Test metadata delete, copy, paste
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.objects
import sigima.params
import sigima.proc.image as sigima_image
import sigima.proc.signal as sigima_signal
from sigima.tests.data import create_paracetamol_signal

from datalab.adapters_metadata.geometry_adapter import GeometryAdapter
from datalab.env import execenv
from datalab.gui.panel.base import BaseDataPanel, PasteMetadataParam
from datalab.gui.panel.image import ImagePanel
from datalab.gui.panel.signal import SignalPanel
from datalab.tests import datalab_test_app_context
from datalab.tests.features.image.roi_app_test import create_test_image_with_roi


def __run_signal_computations(panel: SignalPanel):
    """Test all signal features related to ROI"""
    execenv.print("  Signal features")
    panel.processor.run_feature(sigima_signal.fwhm, sigima.params.FWHMParam())
    panel.processor.run_feature(sigima_signal.fw1e2)


def __run_image_computations(panel: ImagePanel):
    """Test all image features related to ROI"""
    execenv.print("  Image features")
    panel.processor.run_feature(sigima_image.centroid)
    panel.processor.run_feature(sigima_image.enclosing_circle)
    panel.processor.run_feature(
        sigima_image.peak_detection, sigima.params.Peak2DDetectionParam()
    )


def __test_metadata_features(panel: BaseDataPanel):
    """Test all metadata features"""
    # Duplicate the first object
    panel.duplicate_object()

    # Delete metadata of the first object
    for keep_roi in (True, False):  # Test both cases (coverage test)
        panel.delete_metadata(keep_roi=keep_roi)

    # Select and copy metadata of the second object
    panel.objview.select_objects([2])
    source_obj = panel.objview.get_sel_objects()[0]
    source_metadata = source_obj.metadata.copy()

    # Verify source object has geometry results
    geometry_keys = [
        k for k, v in source_metadata.items() if GeometryAdapter.match(k, v)
    ]
    execenv.print(f"  Source object has {len(geometry_keys)} geometry metadata keys")
    assert len(geometry_keys) > 0, "Source object should have geometry results"

    # Copy metadata
    panel.copy_metadata()

    # Select and paste metadata to the first object
    panel.objview.select_objects([1])
    target_obj = panel.objview.get_sel_objects()[0]

    # Verify target has no geometry metadata before paste
    target_geo_keys_before = [
        k for k, v in target_obj.metadata.items() if GeometryAdapter.match(k, v)
    ]
    execenv.print(
        f"  Target object has {len(target_geo_keys_before)} geometry keys before paste"
    )

    # Paste metadata (with default parameters - keep everything)
    param = PasteMetadataParam("Test paste")
    param.keep_geometry = True
    param.keep_tables = True
    param.keep_other = True
    param.keep_roi = True
    panel.paste_metadata(param)

    # Verify the paste worked
    target_metadata_after = target_obj.metadata.copy()
    target_geo_keys_after = [
        k for k, v in target_metadata_after.items() if GeometryAdapter.match(k, v)
    ]
    execenv.print(
        f"  Target object has {len(target_geo_keys_after)} geometry keys after paste"
    )

    # Check that geometry metadata was actually pasted
    assert len(target_geo_keys_after) > 0, (
        "Target object should have geometry results after paste"
    )

    # Verify that all geometry metadata uses the new unified format (_dict)
    for key in target_geo_keys_after:
        # Verify the geometry data is valid
        geometry_data = target_metadata_after[key]
        assert isinstance(geometry_data, dict), f"Geometry data should be dict: {key}"
        assert "title" in geometry_data, f"Missing title in geometry data: {key}"
        assert "coords" in geometry_data, f"Missing coords in geometry data: {key}"
        execenv.print(f"  ✓ Valid geometry entry: {key}")

    execenv.print("  ✓ Metadata copy/paste verification passed")


def test_metadata_app():
    """Run metadata application test scenario"""
    size = 200
    with datalab_test_app_context() as win:
        execenv.print("Metadata application test:")
        # === Signal metadata features test ===
        panel = win.signalpanel
        sig = create_paracetamol_signal(size)
        sig.roi = sigima.objects.create_signal_roi([[26, 41], [125, 146]], indices=True)
        panel.add_object(sig)
        __run_signal_computations(panel)
        __test_metadata_features(panel)
        # === Image metadata features test ===
        panel = win.imagepanel
        param = sigima.objects.NewImageParam.create(height=size, width=size)
        ima = create_test_image_with_roi(param)
        panel.add_object(ima)
        __run_image_computations(panel)
        __test_metadata_features(panel)
        execenv.print("==> OK")


if __name__ == "__main__":
    test_metadata_app()
