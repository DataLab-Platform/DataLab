# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signal ROI manipulation application test (copy/paste, import/export)"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

from sigima.io import read_roi
from sigima.objects import create_signal_roi
from sigima.tests.data import create_paracetamol_signal

from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context

if TYPE_CHECKING:
    from datalab.gui.panel.signal import SignalPanel

SIZE = 200

# Signal ROIs:
SROI1 = [26, 41]
SROI2 = [125, 146]
SROI3 = [60, 80]


def test_signal_roi_copy_paste():
    """Test signal ROI copy and paste functionality"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Signal ROI Copy/Paste test:")
        panel: SignalPanel = win.signalpanel

        # Create first signal with ROI
        sig1 = create_paracetamol_signal(SIZE)
        sig1.title = "Signal with ROI"
        roi1 = create_signal_roi([SROI1, SROI2], indices=True)
        sig1.roi = roi1
        panel.add_object(sig1)

        # Create second signal without ROI
        sig2 = create_paracetamol_signal(SIZE)
        sig2.title = "Signal without ROI"
        panel.add_object(sig2)

        # Create third signal without ROI
        sig3 = create_paracetamol_signal(SIZE)
        sig3.title = "Signal without ROI 2"
        panel.add_object(sig3)

        execenv.print("  Initial state:")
        execenv.print(f"    Signal 1 ROI: {sig1.roi is not None}")
        execenv.print(f"    Signal 2 ROI: {sig2.roi is not None}")
        execenv.print(f"    Signal 3 ROI: {sig3.roi is not None}")

        # Select first signal and copy its ROI
        panel.objview.set_current_item_id(get_uuid(sig1))
        panel.copy_roi()
        execenv.print("  Copied ROI from Signal 1")

        # Select second signal and paste ROI
        panel.objview.set_current_item_id(get_uuid(sig2))
        panel.paste_roi()
        execenv.print("  Pasted ROI to Signal 2")

        # Verify that sig2 now has the same ROI as sig1
        assert sig2.roi is not None, "Signal 2 should have ROI after paste"
        assert len(sig2.roi) == len(sig1.roi), "ROI should have same number of regions"
        execenv.print(f"    Signal 2 now has {len(sig2.roi)} ROI regions")

        # Select third signal and paste ROI (should create new ROI)
        panel.objview.set_current_item_id(get_uuid(sig3))
        panel.paste_roi()
        execenv.print("  Pasted ROI to Signal 3")

        assert sig3.roi is not None, "Signal 3 should have ROI after paste"
        assert len(sig3.roi) == len(sig1.roi), "ROI should have same number of regions"
        execenv.print(f"    Signal 3 now has {len(sig3.roi)} ROI regions")

        # Test pasting to signal that already has ROI (should combine)
        panel.objview.set_current_item_id(get_uuid(sig2))
        panel.copy_roi()
        execenv.print("  Copied ROI from Signal 2")

        # Add a different ROI to sig1
        roi_new = create_signal_roi([SROI3], indices=True)
        sig1.roi = sig1.roi.combine_with(roi_new)
        original_roi_count = len(sig1.roi)
        execenv.print(f"    Signal 1 now has {original_roi_count} ROI regions")

        # Paste the ROI from sig2 into sig1 (should combine)
        panel.objview.set_current_item_id(get_uuid(sig1))
        panel.paste_roi()
        execenv.print("  Pasted ROI to Signal 1 (should combine)")

        # Get fresh reference to sig1 from panel
        sig1_updated = panel.objmodel[get_uuid(sig1)]
        assert sig1_updated.roi is not None, "Signal 1 should still have ROI"
        # After combining, sig1 should have more regions than before
        assert len(sig1_updated.roi) >= original_roi_count, (
            f"Expected at least {original_roi_count} ROI regions, "
            f"got {len(sig1_updated.roi)}"
        )
        execenv.print(
            f"    Signal 1 now has {len(sig1_updated.roi)} ROI regions (combined)"
        )

        execenv.print("  ✓ Signal ROI copy/paste test passed")


def test_signal_roi_copy_paste_multiple_selection():
    """Test signal ROI paste to multiple selected signals"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Signal ROI Copy/Paste with multiple selection test:")
        panel: SignalPanel = win.signalpanel

        # Create source signal with ROI
        sig_src = create_paracetamol_signal(SIZE)
        sig_src.title = "Source with ROI"
        roi = create_signal_roi([SROI1, SROI2], indices=True)
        sig_src.roi = roi
        panel.add_object(sig_src)

        # Create multiple target signals without ROI
        target_signals = []
        for i in range(3):
            sig = create_paracetamol_signal(SIZE)
            sig.title = f"Target signal {i + 1}"
            panel.add_object(sig)
            target_signals.append(sig)

        execenv.print(f"  Created {len(target_signals)} target signals")

        # Copy ROI from source
        panel.objview.set_current_item_id(get_uuid(sig_src))
        panel.copy_roi()
        execenv.print("  Copied ROI from source signal")

        # Select all target signals
        target_uuids = [get_uuid(sig) for sig in target_signals]
        panel.objview.set_current_item_id(target_uuids[0])
        for uuid in target_uuids[1:]:
            panel.objview.set_current_item_id(uuid, extend=True)

        execenv.print(f"  Selected {len(target_uuids)} target signals")

        # Paste ROI to all selected signals
        panel.paste_roi()
        execenv.print("  Pasted ROI to all selected signals")

        # Verify all target signals have ROI
        for i, sig in enumerate(target_signals):
            assert sig.roi is not None, f"Target signal {i + 1} should have ROI"
            assert len(sig.roi) == len(sig_src.roi), (
                f"Target signal {i + 1} should have {len(sig_src.roi)} ROI regions"
            )
            execenv.print(f"    Target signal {i + 1}: {len(sig.roi)} ROI regions ✓")

        execenv.print("  ✓ Multiple selection paste test passed")


def test_signal_roi_import_export():
    """Test signal ROI import and export to/from file functionality"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Signal ROI Import/Export test:")
        panel: SignalPanel = win.signalpanel

        # Create first signal with ROI
        sig1 = create_paracetamol_signal(SIZE)
        sig1.title = "Signal with ROI"
        roi1 = create_signal_roi([SROI1, SROI2, SROI3], indices=True)
        sig1.roi = roi1
        panel.add_object(sig1)

        original_roi_count = len(sig1.roi)
        execenv.print(f"  Signal 1 has {original_roi_count} ROI regions")

        # Export ROI to file
        roi_file = tempfile.mktemp(suffix=".dlabroi")
        try:
            execenv.print("  Exporting ROI to temporary file")

            # Select first signal and export its ROI
            panel.objview.set_current_item_id(get_uuid(sig1))
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

            # Create second signal without ROI
            sig2 = create_paracetamol_signal(SIZE)
            sig2.title = "Signal without ROI"
            panel.add_object(sig2)
            assert sig2.roi is None, "Signal 2 should not have ROI initially"

            # Import ROI from file to second signal
            panel.objview.set_current_item_id(get_uuid(sig2))
            panel.import_roi_from_file(roi_file)
            execenv.print("  Imported ROI to Signal 2")

            # Get fresh reference to sig2 from panel
            sig2_updated = panel.objmodel[get_uuid(sig2)]
            assert sig2_updated.roi is not None, "Signal 2 should have ROI after import"
            assert len(sig2_updated.roi) == original_roi_count, (
                f"Imported ROI should have {original_roi_count} regions"
            )
            execenv.print(f"  ✓ Signal 2 now has {len(sig2_updated.roi)} ROI regions")

            # Test importing ROI to signal that already has ROI (should combine)
            sig3 = create_paracetamol_signal(SIZE)
            sig3.title = "Signal with existing ROI"
            roi3 = create_signal_roi([[90, 110]], indices=True)
            sig3.roi = roi3
            panel.add_object(sig3)
            initial_roi_count = len(sig3.roi)
            execenv.print(f"  Signal 3 has {initial_roi_count} ROI region initially")

            # Import ROI (should combine with existing)
            panel.objview.set_current_item_id(get_uuid(sig3))
            panel.import_roi_from_file(roi_file)
            execenv.print("  Imported ROI to Signal 3 (should combine)")

            # Get fresh reference to sig3 from panel
            sig3_updated = panel.objmodel[get_uuid(sig3)]
            assert sig3_updated.roi is not None, "Signal 3 should still have ROI"
            # After combining, should have more regions
            assert len(sig3_updated.roi) >= initial_roi_count, (
                f"Expected at least {initial_roi_count} ROI regions, "
                f"got {len(sig3_updated.roi)}"
            )
            execenv.print(
                f"  ✓ Signal 3 now has {len(sig3_updated.roi)} ROI regions (combined)"
            )
        finally:
            # Clean up temporary file
            if os.path.exists(roi_file):
                try:
                    os.unlink(roi_file)
                except (PermissionError, OSError):
                    pass  # Ignore cleanup errors on Windows

        execenv.print("  ✓ Signal ROI import/export test passed")


if __name__ == "__main__":
    test_signal_roi_copy_paste()
    test_signal_roi_copy_paste_multiple_selection()
    test_signal_roi_import_export()
