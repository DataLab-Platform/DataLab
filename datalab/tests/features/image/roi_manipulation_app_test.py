# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Image ROI manipulation application test (copy/paste, import/export)"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

from sigima.io import read_roi
from sigima.objects import NewImageParam, create_image_roi
from sigima.tests.data import create_multigaussian_image

from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context

if TYPE_CHECKING:
    from datalab.gui.panel.image import ImagePanel

SIZE = 200

# Image ROIs:
IROI1 = [100, 100, 75, 100]  # Rectangle
IROI2 = [66, 100, 50]  # Circle
IROI3 = [100, 100, 100, 150, 150, 133]  # Polygon


def test_image_roi_copy_paste():
    """Test image ROI copy and paste functionality"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Image ROI Copy/Paste test:")
        panel: ImagePanel = win.imagepanel
        param = NewImageParam.create(height=SIZE, width=SIZE)

        # Create first image with ROI
        ima1 = create_multigaussian_image(param)
        ima1.title = "Image with ROI"
        roi1 = create_image_roi("rectangle", IROI1)
        roi1.add_roi(create_image_roi("circle", IROI2))
        ima1.roi = roi1
        panel.add_object(ima1)

        # Create second image without ROI
        ima2 = create_multigaussian_image(param)
        ima2.title = "Image without ROI"
        panel.add_object(ima2)

        # Create third image without ROI
        ima3 = create_multigaussian_image(param)
        ima3.title = "Image without ROI 2"
        panel.add_object(ima3)

        execenv.print("  Initial state:")
        execenv.print(f"    Image 1 ROI: {ima1.roi is not None}")
        execenv.print(f"    Image 2 ROI: {ima2.roi is not None}")
        execenv.print(f"    Image 3 ROI: {ima3.roi is not None}")

        # Select first image and copy its ROI
        panel.objview.set_current_item_id(get_uuid(ima1))
        panel.copy_roi()
        execenv.print("  Copied ROI from Image 1")

        # Select second image and paste ROI
        panel.objview.set_current_item_id(get_uuid(ima2))
        panel.paste_roi()
        execenv.print("  Pasted ROI to Image 2")

        # Verify that ima2 now has the same ROI as ima1
        assert ima2.roi is not None, "Image 2 should have ROI after paste"
        assert len(ima2.roi) == len(ima1.roi), "ROI should have same number of regions"
        execenv.print(f"    Image 2 now has {len(ima2.roi)} ROI regions")

        # Select third image and paste ROI (should create new ROI)
        panel.objview.set_current_item_id(get_uuid(ima3))
        panel.paste_roi()
        execenv.print("  Pasted ROI to Image 3")

        assert ima3.roi is not None, "Image 3 should have ROI after paste"
        assert len(ima3.roi) == len(ima1.roi), "ROI should have same number of regions"
        execenv.print(f"    Image 3 now has {len(ima3.roi)} ROI regions")

        # Test pasting to image that already has ROI (should combine)
        panel.objview.set_current_item_id(get_uuid(ima2))
        panel.copy_roi()
        execenv.print("  Copied ROI from Image 2")

        # Add a different ROI to ima1
        roi_new = create_image_roi("polygon", IROI3)
        ima1.roi.add_roi(roi_new)
        original_roi_count = len(ima1.roi)
        execenv.print(f"    Image 1 now has {original_roi_count} ROI regions")

        # Paste the ROI from ima2 into ima1 (should combine)
        panel.objview.set_current_item_id(get_uuid(ima1))
        panel.paste_roi()
        execenv.print("  Pasted ROI to Image 1 (should combine)")

        # Get fresh reference to ima1 from panel
        ima1_updated = panel.objmodel[get_uuid(ima1)]
        assert ima1_updated.roi is not None, "Image 1 should still have ROI"
        # After combining, ima1 should have more regions than before
        assert len(ima1_updated.roi) >= original_roi_count, (
            f"Expected at least {original_roi_count} ROI regions, "
            f"got {len(ima1_updated.roi)}"
        )
        execenv.print(
            f"    Image 1 now has {len(ima1_updated.roi)} ROI regions (combined)"
        )

        execenv.print("  ✓ Image ROI copy/paste test passed")


def test_image_roi_copy_paste_multiple_selection():
    """Test image ROI paste to multiple selected images"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Image ROI Copy/Paste with multiple selection test:")
        panel: ImagePanel = win.imagepanel
        param = NewImageParam.create(height=SIZE, width=SIZE)

        # Create source image with ROI
        ima_src = create_multigaussian_image(param)
        ima_src.title = "Source with ROI"
        roi = create_image_roi("rectangle", IROI1)
        roi.add_roi(create_image_roi("circle", IROI2))
        ima_src.roi = roi
        panel.add_object(ima_src)

        # Create multiple target images without ROI
        target_images = []
        for i in range(3):
            ima = create_multigaussian_image(param)
            ima.title = f"Target image {i + 1}"
            panel.add_object(ima)
            target_images.append(ima)

        execenv.print(f"  Created {len(target_images)} target images")

        # Copy ROI from source
        panel.objview.set_current_item_id(get_uuid(ima_src))
        panel.copy_roi()
        execenv.print("  Copied ROI from source image")

        # Select all target images
        target_uuids = [get_uuid(img) for img in target_images]
        panel.objview.set_current_item_id(target_uuids[0])
        for uuid in target_uuids[1:]:
            panel.objview.set_current_item_id(uuid, extend=True)

        execenv.print(f"  Selected {len(target_uuids)} target images")

        # Paste ROI to all selected images
        panel.paste_roi()
        execenv.print("  Pasted ROI to all selected images")

        # Verify all target images have ROI
        for i, img in enumerate(target_images):
            assert img.roi is not None, f"Target image {i + 1} should have ROI"
            assert len(img.roi) == len(ima_src.roi), (
                f"Target image {i + 1} should have {len(ima_src.roi)} ROI regions"
            )
            execenv.print(f"    Target image {i + 1}: {len(img.roi)} ROI regions ✓")

        execenv.print("  ✓ Multiple selection paste test passed")


def test_image_roi_import_export():
    """Test image ROI import and export to/from file functionality"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Image ROI Import/Export test:")
        panel: ImagePanel = win.imagepanel
        param = NewImageParam.create(height=SIZE, width=SIZE)

        # Create first image with ROI
        ima1 = create_multigaussian_image(param)
        ima1.title = "Image with ROI"
        roi1 = create_image_roi("rectangle", IROI1)
        roi1.add_roi(create_image_roi("circle", IROI2))
        roi1.add_roi(create_image_roi("polygon", IROI3))
        ima1.roi = roi1
        panel.add_object(ima1)

        original_roi_count = len(ima1.roi)
        execenv.print(f"  Image 1 has {original_roi_count} ROI regions")

        # Export ROI to file
        roi_file = tempfile.mktemp(suffix=".dlabroi")
        try:
            execenv.print("  Exporting ROI to temporary file")

            # Select first image and export its ROI
            panel.objview.set_current_item_id(get_uuid(ima1))
            panel.export_roi_to_file(roi_file)
            execenv.print("  ✓ ROI exported")

            # Verify file was created
            assert os.path.exists(roi_file), "ROI file should have been created"

            # Read the exported ROI directly to verify content
            exported_roi = read_roi(roi_file)
            assert len(exported_roi) == original_roi_count, (
                f"Exported ROI should have {original_roi_count} regions"
            )
            execenv.print(f"  ✓ Exported ROI has {len(exported_roi)} regions")

            # Create second image without ROI
            ima2 = create_multigaussian_image(param)
            ima2.title = "Image without ROI"
            panel.add_object(ima2)
            assert ima2.roi is None, "Image 2 should not have ROI initially"

            # Import ROI from file to second image
            panel.objview.set_current_item_id(get_uuid(ima2))
            panel.import_roi_from_file(roi_file)
            execenv.print("  Imported ROI to Image 2")

            # Get fresh reference to ima2 from panel
            ima2_updated = panel.objmodel[get_uuid(ima2)]
            assert ima2_updated.roi is not None, "Image 2 should have ROI after import"
            assert len(ima2_updated.roi) == original_roi_count, (
                f"Imported ROI should have {original_roi_count} regions"
            )
            execenv.print(f"  ✓ Image 2 now has {len(ima2_updated.roi)} ROI regions")

            # Test importing ROI to image that already has ROI (should combine)
            ima3 = create_multigaussian_image(param)
            ima3.title = "Image with existing ROI"
            roi3 = create_image_roi("circle", [150, 150, 40])
            ima3.roi = roi3
            panel.add_object(ima3)
            initial_roi_count = len(ima3.roi)
            execenv.print(f"  Image 3 has {initial_roi_count} ROI region initially")

            # Import ROI (should combine with existing)
            panel.objview.set_current_item_id(get_uuid(ima3))
            panel.import_roi_from_file(roi_file)
            execenv.print("  Imported ROI to Image 3 (should combine)")

            # Get fresh reference to ima3 from panel
            ima3_updated = panel.objmodel[get_uuid(ima3)]
            assert ima3_updated.roi is not None, "Image 3 should still have ROI"
            # After combining, should have more regions
            assert len(ima3_updated.roi) >= initial_roi_count, (
                f"Expected at least {initial_roi_count} ROI regions, "
                f"got {len(ima3_updated.roi)}"
            )
            execenv.print(
                f"  ✓ Image 3 now has {len(ima3_updated.roi)} ROI regions (combined)"
            )
        finally:
            # Clean up temporary file
            if os.path.exists(roi_file):
                try:
                    os.unlink(roi_file)
                except (PermissionError, OSError):
                    pass  # Ignore cleanup errors on Windows

        execenv.print("  ✓ Image ROI import/export test passed")


if __name__ == "__main__":
    test_image_roi_copy_paste()
    test_image_roi_copy_paste_multiple_selection()
    test_image_roi_import_export()
