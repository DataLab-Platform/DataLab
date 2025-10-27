# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Processing metadata storage tests

Tests that ALL processing patterns (1-to-1, 2-to-1, n-to-1) correctly
store metadata in object metadata options, regardless of whether they
support interactive re-processing.

This is an integration test that verifies the metadata storage infrastructure
works correctly across all computation patterns.

Note: interactive_processing_test.py tests the interactive re-processing UI feature.
"""

# guitest: show

import guidata.dataset as gds
from guidata.qthelpers import qt_app_context
from sigima.proc.signal.filtering import GaussianParam

from datalab.gui.processor.base import PROCESSING_PARAMETERS_OPTION
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def test_metadata_all_patterns():
    """Test that metadata is stored for all processing patterns."""
    with qt_app_context():
        with datalab_test_app_context() as win:
            # Get panels
            sig_panel = win.signalpanel
            img_panel = win.imagepanel

            # === Test 1: 1-to-1 pattern (with parameters) ===
            sig_panel.new_object()
            sig1 = sig_panel.objmodel.get_all_objects()[-1]

            param = GaussianParam.create(sigma=2.0)
            sig_panel.processor.compute_1_to_1(
                sig_panel.processor.get_feature("gaussian_filter").function,
                param=param,
                title="Gaussian filter",
            )
            filtered_sig = sig_panel.objmodel.get_all_objects()[-1]

            # Check metadata contains all keys including pattern type
            assert PROCESSING_PARAMETERS_OPTION in filtered_sig.get_metadata_options()
            option_dict = filtered_sig.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
            assert option_dict["func_name"] == "gaussian_filter"
            assert option_dict["pattern"] == "1-to-1"
            assert option_dict["source_uuid"] == get_uuid(sig1)
            assert option_dict["param_json"] == gds.dataset_to_json(param)

            # === Test 2: 2-to-1 pattern (single operand) ===
            sig_panel.new_object()
            sig2 = sig_panel.objmodel.get_all_objects()[-1]
            sig_panel.objview.select_objects([sig1])  # Select only sig1
            sig_panel.processor.compute_2_to_1(
                obj2=sig2,
                obj2_name="signal to subtract",
                func=sig_panel.processor.get_feature("difference").function,
                title="Difference",
            )
            subtracted_sig = sig_panel.objmodel.get_all_objects()[-1]

            # Check lightweight metadata (no params, but has pattern and sources)
            assert PROCESSING_PARAMETERS_OPTION in subtracted_sig.get_metadata_options()
            option_dict = subtracted_sig.get_metadata_option(
                PROCESSING_PARAMETERS_OPTION
            )
            assert option_dict["pattern"] == "2-to-1"
            assert option_dict["source_uuids"] == [get_uuid(sig1), get_uuid(sig2)]
            assert len(option_dict["source_uuids"]) == 2
            # Should NOT have full parameters stored
            assert "param" not in option_dict

            # === Test 3: n-to-1 pattern (single operand mode) ===
            img_panel.new_object()
            img1 = img_panel.objmodel.get_all_objects()[-1]
            img_panel.new_object()
            img2 = img_panel.objmodel.get_all_objects()[-1]
            img_panel.objview.select_objects([img1, img2])
            img_panel.processor.compute_n_to_1(
                func=img_panel.processor.get_feature("addition").function,
                title="Addition",
            )
            sum_img = img_panel.objmodel.get_all_objects()[-1]

            # Check lightweight metadata
            assert PROCESSING_PARAMETERS_OPTION in sum_img.get_metadata_options()
            option_dict = sum_img.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
            assert option_dict["pattern"] == "n-to-1"
            assert "source_uuids" in option_dict
            assert len(option_dict["source_uuids"]) == 2
            # Should NOT have full parameters stored
            assert "param" not in option_dict

            print("✓ All processing patterns store appropriate metadata")


if __name__ == "__main__":
    test_metadata_all_patterns()
