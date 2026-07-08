# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Analysis parameters edit unit test
----------------------------------

Test the editable "Analysis" tab of the Object Properties widget.

This verifies that a 1-to-0 analysis operation (e.g. 2D peak detection) can be
re-run in place with modified parameters through
:meth:`ObjectProp.setup_analysis_tab` / :meth:`ObjectProp.apply_analysis_parameters`.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import sigima.params as sigima_param
from sigima.tests.data import create_peak_image

from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.processor.base import extract_analysis_parameters
from datalab.tests import datalab_test_app_context


def test_analysis_parameters_edit_image():
    """Test editing and re-running a 1-to-0 analysis via the Analysis tab."""
    with datalab_test_app_context(console=False) as win:
        execenv.print("Analysis parameters edit test (image peak detection):")
        panel = win.imagepanel

        # Create a multi-peak image guaranteed to yield detections
        img = create_peak_image()
        panel.add_object(img)

        # Run 2D peak detection: a 1-to-0 analysis with an editable parameter
        det_param = sigima_param.Peak2DDetectionParam.create(
            create_rois=False, threshold=0.5
        )
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("peak_detection", det_param)

        # The analysis parameters must be stored as a single 1-to-0 dataset
        proc_params = extract_analysis_parameters(img)
        assert proc_params is not None, "Analysis parameters should be stored"
        assert proc_params.pattern == "1-to-0"
        assert proc_params.param is not None
        assert not isinstance(proc_params.param, list)
        assert proc_params.param.threshold == 0.5
        execenv.print("  ✓ Analysis parameters stored (threshold=0.5)")

        # Set up the editable Analysis tab
        objprop = panel.objprop
        assert objprop.setup_analysis_tab(img) is True
        assert objprop.analysis_param_editor is not None
        execenv.print("  ✓ Analysis tab set up")

        # Modify the threshold and (deliberately) enable ROI creation to verify
        # the create_rois guard forces it back to False on apply
        objprop.analysis_param_editor.dataset.threshold = 0.8
        objprop.analysis_param_editor.dataset.create_rois = True

        # Apply: re-run the analysis in place with the modified parameters
        objprop.apply_analysis_parameters(img)

        # The stored analysis parameters must reflect the new threshold
        proc_params2 = extract_analysis_parameters(img)
        assert proc_params2 is not None
        assert proc_params2.param.threshold == 0.8, "Threshold change must be applied"
        execenv.print("  ✓ Analysis re-ran with new threshold (0.8)")

        # ROI guard: create_rois must have been forced to False (no ROI created)
        assert proc_params2.param.create_rois is False, (
            "create_rois must be forced to False on re-analysis"
        )
        assert not img.roi, "No ROI should be created on re-analysis"
        execenv.print("  ✓ ROI creation guard held (create_rois=False)")


if __name__ == "__main__":
    test_analysis_parameters_edit_image()
