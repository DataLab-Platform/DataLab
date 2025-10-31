# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result label deletion test:

Test that when a result label is deleted from the plot, the associated result
metadata is also removed and doesn't reappear when re-selecting the signal.

This is a regression test for the bug where:
1. Computing FWHM adds a result label
2. Deleting the label doesn't remove the result metadata
3. De-selecting and re-selecting the signal makes the label reappear
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.params
import sigima.proc.signal as sips
from sigima.tests.data import create_paracetamol_signal

from datalab.adapters_metadata.geometry_adapter import GeometryAdapter
from datalab.env import execenv
from datalab.tests import datalab_test_app_context


def test_result_label_deletion() -> None:
    """Test that deleting a result label also removes the result metadata"""
    with datalab_test_app_context() as win:
        panel = win.signalpanel

        # Create a Gaussian curve
        execenv.print("Creating paracetamol signal...")
        sig = create_paracetamol_signal()
        panel.add_object(sig)
        execenv.print(f"  Added signal: {sig.title}")

        # Compute the FWHM (this adds a result to metadata)
        execenv.print("Computing FWHM...")
        panel.processor.run_feature(sips.fwhm, sigima.params.FWHMParam())

        # Check that the result metadata exists
        sig = panel.objview.get_sel_objects()[0]
        geometry_keys_before = [
            k for k, v in sig.metadata.items() if GeometryAdapter.match(k, v)
        ]
        execenv.print(f"  Geometry metadata keys before: {len(geometry_keys_before)}")
        assert len(geometry_keys_before) > 0, "Should have geometry result metadata"

        # Get the plot and find the result label item
        plot = panel.plothandler.plot
        label_items = [item for item in plot.items if hasattr(item, "labelparam")]
        execenv.print(f"  Found {len(label_items)} label items on plot")

        # Find the FWHM result label (it should be in the result_label_to_adapter)
        result_label = None
        for item in label_items:
            # Check if this item is tracked as a result label
            if item in panel.plothandler._BasePlotHandler__result_label_to_adapter:
                result_label = item
                break

        assert result_label is not None, "Should find the FWHM result label"
        execenv.print(f"  Found result label: {result_label.title()}")

        # Delete the label from the plot (simulating user action)
        execenv.print("Deleting result label from plot...")
        plot.del_item(result_label)

        # NOTE: The SIG_ITEM_REMOVED signal is only emitted when items are removed
        # via del_items (used in remove_all_shape_items), not via del_item.
        # So we need to force a refresh to trigger the removal handler.
        panel.plothandler.refresh_plot("selected")

        # Check that the result metadata has been removed
        sig = panel.objview.get_sel_objects()[0]
        geometry_keys_after_delete = [
            k for k, v in sig.metadata.items() if GeometryAdapter.match(k, v)
        ]
        execenv.print(
            f"  Geometry metadata keys after delete: {len(geometry_keys_after_delete)}"
        )
        assert len(geometry_keys_after_delete) == 0, "Result metadata should be removed"

        # Create another signal and select it (to deselect the first one)
        execenv.print("Creating second signal to deselect first...")
        sig2 = create_paracetamol_signal()
        sig2.title = "Paracetamol 2"
        panel.add_object(sig2)

        # Re-select the first signal
        execenv.print("Re-selecting first signal...")
        panel.objview.select_objects([1])

        # Check that the result metadata is still absent and label doesn't reappear
        sig = panel.objview.get_sel_objects()[0]
        geometry_keys_after_reselect = [
            k for k, v in sig.metadata.items() if GeometryAdapter.match(k, v)
        ]
        execenv.print(
            f"  Geometry metadata keys after reselect: "
            f"{len(geometry_keys_after_reselect)}"
        )
        assert len(geometry_keys_after_reselect) == 0, (
            "Result metadata should still be absent after re-selection"
        )

        # Check that the label doesn't reappear on the plot
        label_items_after = [item for item in plot.items if hasattr(item, "labelparam")]
        result_labels_after = [
            item
            for item in label_items_after
            if item in panel.plothandler._BasePlotHandler__result_items_mapping
        ]
        execenv.print(
            f"  Result labels on plot after reselect: {len(result_labels_after)}"
        )
        assert len(result_labels_after) == 0, (
            "Result label should not reappear after re-selection"
        )

        execenv.print("✓ Test passed: Result label deletion works correctly")


if __name__ == "__main__":
    test_result_label_deletion()
