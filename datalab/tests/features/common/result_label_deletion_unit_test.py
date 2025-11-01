# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Merged result label test:

Test that multiple results are merged into a single label that displays all results.

This test verifies that:
1. Computing FWHM adds a result to the merged label
2. Computing FW1e2 adds another result to the same merged label
3. The merged label contains both results
4. The merged label is read-only (cannot be deleted to remove individual results)
5. De-selecting and re-selecting the signal maintains the merged label
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.params
import sigima.proc.signal as sips
from sigima.tests.data import create_paracetamol_signal

from datalab.adapters_metadata.geometry_adapter import GeometryAdapter
from datalab.env import execenv
from datalab.tests import datalab_test_app_context


def test_merged_result_label() -> None:
    """Test that multiple results are merged into a single label"""
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

        # Force refresh to ensure shapes are added
        panel.plothandler.refresh_plot("selected", force=True)

        # Check that the result metadata exists
        sig = panel.objview.get_sel_objects()[0]
        geometry_results_1 = list(GeometryAdapter.iterate_from_obj(sig))
        execenv.print(f"  Geometry results after FWHM: {len(geometry_results_1)}")
        assert len(geometry_results_1) == 1, "Should have one geometry result"

        # Get the plot and find the merged result label item
        plot = panel.plothandler.plot
        label_items = [item for item in plot.items if hasattr(item, "labelparam")]

        # Find the merged result label - it should be on the plot
        merged_label = None
        for item in label_items:
            if hasattr(item, "is_readonly") and item.is_readonly():
                merged_label = item
                break

        assert merged_label is not None, "Should find the merged result label"
        execenv.print(f"  Found merged result label: {merged_label.title()}")

        # Check that the label is read-only
        assert merged_label.is_readonly(), "Merged label should be read-only"
        execenv.print("  ✓ Merged label is read-only")

        # Get the label text to verify it contains FWHM result
        label_text_1 = merged_label.text_string
        assert "fwhm" in label_text_1.lower(), "Label should contain FWHM result"
        execenv.print("  ✓ Label contains FWHM result")

        # Compute FW1e2 (this adds another result to metadata)
        execenv.print("Computing FW1e2...")
        panel.processor.run_feature(sips.fw1e2)

        # Check that we now have two results
        sig = panel.objview.get_sel_objects()[0]
        geometry_results_2 = list(GeometryAdapter.iterate_from_obj(sig))
        execenv.print(f"  Geometry results after FW1e2: {len(geometry_results_2)}")
        assert len(geometry_results_2) == 2, "Should have two geometry results"

        # Verify the merged label now contains both results
        merged_labels = [
            item
            for item in plot.items
            if hasattr(item, "labelparam")
            and hasattr(item, "is_readonly")
            and item.is_readonly()
        ]

        execenv.print(f"  Merged result labels on plot: {len(merged_labels)}")
        assert len(merged_labels) == 1, "Should have exactly one merged result label"

        merged_label_2 = merged_labels[0]
        label_text_2 = merged_label_2.text_string
        assert "fwhm" in label_text_2.lower(), "Label should contain FWHM result"
        assert "fw1e2" in label_text_2.lower(), "Label should contain FW1e2 result"
        assert "<hr>" in label_text_2, "Should contain separator between results"
        execenv.print("  ✓ Merged label contains both FWHM and FW1e2 results")

        # Create another signal and select it (to deselect the first one)
        execenv.print("Creating second signal to deselect first...")
        sig2 = create_paracetamol_signal()
        sig2.title = "Paracetamol 2"
        panel.add_object(sig2)

        # Re-select the first signal
        execenv.print("Re-selecting first signal...")
        panel.objview.select_objects([1])

        # Check that the merged label still exists and contains both results
        sig = panel.objview.get_sel_objects()[0]
        geometry_results_after = list(GeometryAdapter.iterate_from_obj(sig))
        execenv.print(
            f"  Geometry results after reselect: {len(geometry_results_after)}"
        )
        assert len(geometry_results_after) == 2, (
            "Should still have two geometry results after re-selection"
        )

        # Check that the merged label is still present
        merged_labels_after = [
            item
            for item in plot.items
            if hasattr(item, "labelparam")
            and hasattr(item, "is_readonly")
            and item.is_readonly()
        ]

        execenv.print(
            f"  Merged result labels on plot after reselect: {len(merged_labels_after)}"
        )
        assert len(merged_labels_after) == 1, (
            "Should still have one merged result label after re-selection"
        )

        merged_label_final = merged_labels_after[0]
        label_text_final = merged_label_final.text_string
        assert "fwhm" in label_text_final.lower(), (
            "Label should still contain FWHM result"
        )
        assert "fw1e2" in label_text_final.lower(), (
            "Label should still contain FW1e2 result"
        )
        execenv.print("  ✓ Merged label persists after re-selection")

        execenv.print("✓ Test passed: Merged result label works correctly")


if __name__ == "__main__":
    test_merged_result_label()
