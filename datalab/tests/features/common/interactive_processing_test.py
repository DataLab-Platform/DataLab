# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Interactive re-processing feature tests

Tests the end-to-end interactive processing workflow where users can
modify processing parameters and re-apply them to regenerate the result.

This includes:
- Parameter modification and re-application
- Parameter serialization/deserialization
- Processing metadata storage for 1-to-1 operations
- Handling of operations with and without parameters

Note: metadata_all_patterns_test.py verifies metadata storage for all
processing patterns (1-to-1, 2-to-1, n-to-1).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
from guidata.dataset import json_to_dataset
from guidata.qthelpers import qt_app_context
from sigima.params import GaussianParam, MovingAverageParam

from datalab.gui.processor.base import PROCESSING_PARAMETERS_OPTION
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def test_signal_interactive_processing():
    """Test interactive processing for signals"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            processor = panel.processor

            # Create a test signal
            panel.new_object()
            signal = panel.objview.get_current_object()
            assert signal is not None

            # Apply a Gaussian filter (which has parameters)
            param = GaussianParam.create(sigma=2.0)
            processor.compute_1_to_1(
                processor.get_feature("gaussian_filter").function,
                param=param,
                title="Gaussian filter",
            )

            # Get the filtered signal
            filtered_sig = panel.objview.get_current_object()
            assert filtered_sig is not None
            assert filtered_sig != signal

            # Check that processing metadata was stored
            assert PROCESSING_PARAMETERS_OPTION in filtered_sig.get_metadata_options()
            option_dict = filtered_sig.get_metadata_option(PROCESSING_PARAMETERS_OPTION)

            # Verify metadata content
            assert option_dict["source_uuid"] == get_uuid(signal)
            assert option_dict["func_name"] == "gaussian_filter"

            # Verify the parameter can be deserialized
            stored_param = json_to_dataset(option_dict["param_json"])
            assert stored_param.sigma == 2.0


def test_image_interactive_processing():
    """Test interactive processing for images"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.imagepanel
            processor = panel.processor

            # Create a test image
            panel.new_object()
            image = panel.objview.get_current_object()
            assert image is not None

            # Apply a moving average filter (which has parameters)
            param = MovingAverageParam.create(n=5)
            processor.compute_1_to_1(
                processor.get_feature("moving_average").function,
                param=param,
                title="Moving average",
            )

            # Get the filtered image
            filtered_ima = panel.objview.get_current_object()
            assert filtered_ima is not None
            assert filtered_ima != image

            # Check that processing metadata was stored
            assert PROCESSING_PARAMETERS_OPTION in filtered_ima.get_metadata_options()
            option_dict = filtered_ima.get_metadata_option(PROCESSING_PARAMETERS_OPTION)

            # Verify metadata content
            assert option_dict["source_uuid"] == get_uuid(image)
            assert option_dict["func_name"] == "moving_average"

            # Verify the parameter can be deserialized
            stored_param = json_to_dataset(option_dict["param_json"])
            assert stored_param.n == 5


def test_processing_without_parameters():
    """Test that processing without parameters doesn't store metadata"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            processor = panel.processor

            # Create a test signal
            panel.new_object()
            signal = panel.objview.get_current_object()
            assert signal is not None

            # Apply absolute value (which has NO parameters)
            processor.compute_1_to_1(
                processor.get_feature("absolute").function,
                title="Absolute value",
            )

            # Get the result signal
            result_signal = panel.objview.get_current_object()
            assert result_signal is not None
            assert result_signal != signal

            # Check that processing metadata was NOT stored
            assert PROCESSING_PARAMETERS_OPTION in result_signal.get_metadata_options()
            option_dict = result_signal.get_metadata_option(
                PROCESSING_PARAMETERS_OPTION
            )
            assert "param_json" not in option_dict


def test_recompute():
    """Test recompute feature for signals"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            processor = panel.processor

            # Create a test signal
            panel.new_object()
            signal = panel.objview.get_current_object()
            signal_uuid = get_uuid(signal)

            # Apply a Gaussian filter with initial parameters
            param = GaussianParam.create(sigma=2.0)
            processor.run_feature("gaussian_filter", param=param)
            filtered_sig = panel.objview.get_current_object()
            original_data = filtered_sig.y.copy()

            # Recompute with different input signal data
            constant = 1.23098765
            signal.y += constant
            panel.recompute_processing()

            assert np.allclose(filtered_sig.y, original_data + constant)

            # Verify metadata is correct
            assert PROCESSING_PARAMETERS_OPTION in filtered_sig.get_metadata_options()
            option_dict = filtered_sig.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
            assert option_dict["source_uuid"] == signal_uuid
            assert option_dict["func_name"] == "gaussian_filter"


if __name__ == "__main__":
    test_signal_interactive_processing()
    test_image_interactive_processing()
    test_processing_without_parameters()
    test_recompute()
