# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result deletion unit test
--------------------------

Test the deletion of analysis results from objects.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from sigima.objects import Gauss2DParam, create_image_from_param, create_image_roi
from sigima.tests.data import create_paracetamol_signal

from datalab.adapters_metadata import GeometryAdapter, TableAdapter
from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.processor.base import extract_analysis_parameters
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def test_delete_results_image():
    """Test deletion of analysis results from images"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Image result deletion test:")
        panel = win.imagepanel

        # Create a test image
        param = Gauss2DParam.create(height=200, width=200, sigma=20)
        img = create_image_from_param(param)
        panel.add_object(img)

        # Run centroid analysis to create results
        execenv.print("  Running centroid analysis...")
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("centroid")

        # Verify that results exist
        img_refreshed = panel.objmodel[get_uuid(img)]
        adapter_before = GeometryAdapter.from_obj(img_refreshed, "centroid")
        assert adapter_before is not None, (
            "Centroid result should exist before deletion"
        )
        execenv.print("  ✓ Centroid result created")

        # Delete all results
        execenv.print("  Deleting all results...")
        panel.objview.select_objects([get_uuid(img)])
        panel.delete_results()

        # Verify that results were deleted
        img_after = panel.objmodel[get_uuid(img)]
        adapter_after = GeometryAdapter.from_obj(img_after, "centroid")
        assert adapter_after is None, "Centroid result should be deleted"
        execenv.print("  ✓ Centroid result deleted")


def test_delete_results_signal():
    """Test deletion of analysis results from signals"""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Signal result deletion test:")
        panel = win.signalpanel

        # Create a test signal
        sig = create_paracetamol_signal()
        panel.add_object(sig)

        # Run stats analysis to create table results
        execenv.print("  Running stats analysis...")
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("stats")

        # Verify that results exist
        sig_refreshed = panel.objmodel[get_uuid(sig)]
        tables_before = list(TableAdapter.iterate_from_obj(sig_refreshed))
        assert len(tables_before) > 0, "Stats result should exist before deletion"
        execenv.print("  ✓ Stats result created")

        # Delete all results
        execenv.print("  Deleting all results...")
        panel.objview.select_objects([get_uuid(sig)])
        panel.delete_results()

        # Verify that results were deleted
        sig_after = panel.objmodel[get_uuid(sig)]
        tables_after = list(TableAdapter.iterate_from_obj(sig_after))
        assert len(tables_after) == 0, "Stats result should be deleted"
        execenv.print("  ✓ Stats result deleted")


def test_delete_results_clears_analysis_parameters():
    """Test that deleting results also clears analysis parameters.

    This prevents auto_recompute_analysis from attempting to recompute
    deleted analyses when ROI changes.
    """
    with datalab_test_app_context(console=False) as win:
        execenv.print("Test delete_results clears analysis parameters:")
        panel = win.imagepanel

        # Create a test image
        param = Gauss2DParam.create(height=200, width=200, sigma=20)
        img = create_image_from_param(param)
        panel.add_object(img)

        # Run centroid analysis to create results and store analysis parameters
        execenv.print("  Running centroid analysis...")
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("centroid")

        # Verify that analysis parameters exist
        img_refreshed = panel.objmodel[get_uuid(img)]
        analysis_params = extract_analysis_parameters(img_refreshed)
        assert analysis_params is not None, (
            "Analysis parameters should exist after running centroid"
        )
        assert analysis_params.func_name == "centroid", (
            "Analysis parameters should store the centroid function name"
        )
        execenv.print("  ✓ Analysis parameters stored")

        # Delete all results
        execenv.print("  Deleting all results...")
        panel.objview.select_objects([get_uuid(img)])
        panel.delete_results()

        # Verify that analysis parameters were also cleared
        img_after = panel.objmodel[get_uuid(img)]
        analysis_params_after = extract_analysis_parameters(img_after)
        assert analysis_params_after is None, (
            "Analysis parameters should be cleared after deleting results"
        )
        execenv.print("  ✓ Analysis parameters cleared")

        # Now add a ROI and verify no auto-recompute happens (no new results)
        execenv.print("  Adding ROI to verify no auto-recompute...")
        roi = create_image_roi("rectangle", [25, 25, 100, 100])
        img_after.roi = roi
        panel.processor.auto_recompute_analysis(img_after)

        # Verify that no new results were created
        adapter_after_roi = GeometryAdapter.from_obj(img_after, "centroid")
        assert adapter_after_roi is None, (
            "No centroid result should be created after ROI change "
            "because analysis parameters were cleared"
        )
        execenv.print(
            "  ✓ No auto-recompute after ROI change (analysis params cleared)"
        )
        execenv.print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_delete_results_image()
    test_delete_results_signal()
    test_delete_results_clears_analysis_parameters()
