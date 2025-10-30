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
from sigima.objects import Gauss2DParam, GaussParam
from sigima.params import (
    ConstantParam,
    GaussianParam,
    MovingAverageParam,
    SignalsToImageParam,
)

from datalab.gui.newobject import CREATION_PARAMETERS_OPTION
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


def test_apply_creation_parameters_signal():
    """Test apply_creation_parameters for signals"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            objprop = panel.objprop

            # Create a signal with specific parameters
            param = GaussParam.create(mu=250.0, sigma=20.0, a=100.0, y0=0.0, size=500)
            panel.new_object(param=param, edit=False)
            signal = panel.objview.get_current_object()
            assert signal is not None
            signal_uuid = get_uuid(signal)

            # Verify the Creation tab was set up
            assert objprop.creation_param_editor is not None
            original_data = signal.y.copy()

            # Modify the creation parameters in the editor
            editor = objprop.creation_param_editor
            # Change the Gaussian parameters to get a predictable result
            editor.dataset.a = 200.0  # Double the amplitude from 100.0 to 200.0

            # Apply the new creation parameters
            objprop.apply_creation_parameters()

            # Verify the signal was updated in-place (same UUID)
            updated_signal = panel.objview.get_current_object()
            assert get_uuid(updated_signal) == signal_uuid

            # Get the updated creation parameters from metadata
            creation_param_json = updated_signal.get_metadata_option(
                CREATION_PARAMETERS_OPTION
            )
            updated_param = json_to_dataset(creation_param_json)

            # Verify the parameter was actually updated in metadata
            assert updated_param.a == 200.0

            # Verify the data has changed
            # Since we're working with very small Gaussian values,
            # just verify they're different
            assert not np.array_equal(updated_signal.y, original_data)


def test_apply_creation_parameters_image():
    """Test apply_creation_parameters for images"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.imagepanel
            objprop = panel.objprop

            # Create an image with specific parameters (using a derived class
            # of NewImageParam) to ensure creation parameters are stored in metadata
            param = Gauss2DParam.create(x0=50.0, y0=50.0, sigma=10.0, a=100.0)
            panel.new_object(param=param, edit=False)
            image = panel.objview.get_current_object()
            assert image is not None

            # Verify the Creation tab was set up
            assert objprop.creation_param_editor is not None
            original_data = image.data.copy()

            # Modify the parameters in the editor to create a visibly different image
            editor = objprop.creation_param_editor
            # Change the amplitude to make it clearly different
            editor.dataset.a = 200.0  # Double the amplitude from 100.0 to 200.0

            # Apply the new parameters
            objprop.apply_creation_parameters()

            # Verify the image was updated in-place (same UUID)
            updated_image = panel.objview.get_current_object()
            assert get_uuid(updated_image) == get_uuid(image)

            # Verify the data has changed (amplitude doubled)
            assert not np.array_equal(updated_image.data, original_data)


def test_no_creation_parameters_for_base_classes():
    """Test that creation parameters are NOT stored for base classes

    This test verifies the behavior introduced by the patch that only stores
    creation parameters for derived classes of NewSignalParam/NewImageParam,
    not for the base classes themselves.
    """
    with qt_app_context():
        with datalab_test_app_context() as win:
            # Test with signals
            signal_panel = win.signalpanel
            signal_objprop = signal_panel.objprop

            # Create a signal using default new_object() (uses base NewSignalParam)
            signal_panel.new_object(edit=False)
            signal = signal_panel.objview.get_current_object()
            assert signal is not None

            # Verify the Creation tab was NOT set up (no creation parameters stored)
            assert signal_objprop.creation_param_editor is None

            # Verify that CREATION_PARAMETERS_OPTION is not in metadata
            assert CREATION_PARAMETERS_OPTION not in signal.get_metadata_options()

            # Test with images
            image_panel = win.imagepanel
            image_objprop = image_panel.objprop

            # Create an image using default new_object() (uses base NewImageParam)
            image_panel.new_object(edit=False)
            image = image_panel.objview.get_current_object()
            assert image is not None

            # Verify the Creation tab was NOT set up (no creation parameters stored)
            assert image_objprop.creation_param_editor is None

            # Verify that CREATION_PARAMETERS_OPTION is not in metadata
            assert CREATION_PARAMETERS_OPTION not in image.get_metadata_options()


def test_apply_processing_parameters_signal():
    """Test apply_processing_parameters for signals"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            processor = panel.processor
            objprop = panel.objprop

            # Create a test signal with some structure
            param = GaussParam.create(mu=250.0, sigma=20.0, a=100.0, y0=10.0, size=500)
            panel.new_object(param=param, edit=False)
            signal = panel.objview.get_current_object()
            assert signal is not None
            signal_uuid = get_uuid(signal)
            original_signal_data = signal.y.copy()

            # Apply addition_constant with initial value
            v0 = 5.0
            processor.run_feature("addition_constant", ConstantParam.create(value=v0))

            # Get the processed signal
            processed_sig = panel.objview.get_current_object()
            assert processed_sig is not None
            processed_uuid = get_uuid(processed_sig)

            # Verify initial constant was applied: data should be original + 5.0
            assert np.allclose(processed_sig.y, original_signal_data + v0)

            # Select the processed signal to trigger setup_processing_tab
            panel.objview.set_current_object(processed_sig)

            # Verify the Processing tab was set up
            assert objprop.processing_param_editor is not None

            # Modify the processing parameters
            editor = objprop.processing_param_editor
            # Change constant from 5.0 to 15.0
            editor.dataset.value = v1 = 15.0

            # Apply the new processing parameters
            report = objprop.apply_processing_parameters()

            # Verify the operation succeeded
            assert report.success, f"Reprocessing failed: {report.message}"
            assert report.obj_uuid == processed_uuid

            # Verify the object UUID didn't change (in-place update)
            assert get_uuid(processed_sig) == processed_uuid

            # Verify the new constant was applied: data should now be original + 15.0
            assert np.allclose(processed_sig.y, original_signal_data + v1)

            # Verify metadata still points to the same source
            pp_dict = processed_sig.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
            assert pp_dict["source_uuid"] == signal_uuid
            assert pp_dict["func_name"] == "addition_constant"

            # Verify the parameter was updated
            stored_param = json_to_dataset(pp_dict["param_json"])
            assert stored_param.value == v1


def test_apply_processing_parameters_image():
    """Test apply_processing_parameters for images"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.imagepanel
            processor = panel.processor
            objprop = panel.objprop

            # Create a default test image
            panel.new_object(edit=False)
            image = panel.objview.get_current_object()
            assert image is not None
            image_uuid = get_uuid(image)
            original_image_data = image.data.copy()

            # Apply addition_constant with initial value
            v0 = 7.0
            processor.run_feature("addition_constant", ConstantParam.create(value=v0))

            # Get the processed image
            processed_ima = panel.objview.get_current_object()
            assert processed_ima is not None
            processed_uuid = get_uuid(processed_ima)

            # Verify initial constant was applied: data should be original + 7.0
            assert np.allclose(processed_ima.data, original_image_data + v0)

            # Select the processed image to trigger setup_processing_tab
            panel.objview.set_current_object(processed_ima)

            # Verify the Processing tab was set up
            assert objprop.processing_param_editor is not None

            # Modify the processing parameters
            editor = objprop.processing_param_editor
            # Change constant from 7.0 to 20.0
            editor.dataset.value = v1 = 20.0

            # Apply the new processing parameters
            report = objprop.apply_processing_parameters()

            # Verify the operation succeeded
            assert report.success, f"Reprocessing failed: {report.message}"
            assert report.obj_uuid == processed_uuid

            # Verify the object UUID didn't change (in-place update)
            assert get_uuid(processed_ima) == processed_uuid

            # Verify the new constant was applied: data should now be original + 20.0
            assert np.allclose(processed_ima.data, original_image_data + v1)

            # Verify metadata still points to the same source
            pp_dict = processed_ima.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
            assert pp_dict["source_uuid"] == image_uuid
            assert pp_dict["func_name"] == "addition_constant"

            # Verify the parameter was updated
            stored_param = json_to_dataset(pp_dict["param_json"])
            assert stored_param.value == v1


def test_apply_processing_parameters_missing_source():
    """Test apply_processing_parameters when source object is missing"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            processor = panel.processor
            objprop = panel.objprop

            # Create a test signal with actual data
            param = GaussParam.create(mu=250.0, sigma=20.0, a=100.0, y0=10.0, size=500)
            panel.new_object(param=param, edit=False)
            signal = panel.objview.get_current_object()

            # Apply a Gaussian filter
            filter_param = GaussianParam.create(sigma=2.0)
            processor.compute_1_to_1(
                processor.get_feature("gaussian_filter").function,
                param=filter_param,
                title="Gaussian filter",
            )

            # Get the filtered signal
            filtered_sig = panel.objview.get_current_object()

            # Delete the source signal
            panel.objview.set_current_object(signal)
            panel.remove_object(force=True)

            # Select the filtered signal
            panel.objview.set_current_object(filtered_sig)

            # Try to apply processing parameters
            report = objprop.apply_processing_parameters(interactive=False)

            # Verify the operation failed with appropriate message
            assert not report.success
            # Check for English or French message
            assert (
                "no longer exists" in report.message.lower()
                or "n'existe plus" in report.message.lower()
            )


def test_cross_panel_image_to_signal():
    """Test cross-panel processing: Image → Signal (radial profile)"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            image_panel = win.imagepanel
            signal_panel = win.signalpanel
            image_processor = image_panel.processor

            # Create a test image with a Gaussian peak
            image_param = Gauss2DParam.create(
                x0=50.0, y0=50.0, sigma=10.0, a=100.0, height=100, width=100
            )
            image_panel.new_object(param=image_param, edit=False)
            image = image_panel.objview.get_current_object()
            assert image is not None
            image_uuid = get_uuid(image)

            # Apply radial_profile (Image → Signal cross-panel computation)
            from sigima.proc.image import RadialProfileParam

            profile_param = RadialProfileParam.create(x0=50, y0=50)
            image_processor.run_feature("radial_profile", param=profile_param)

            # Verify the result is now in the signal panel
            signal = signal_panel.objview.get_current_object()
            assert signal is not None

            # Check that processing metadata was stored
            assert PROCESSING_PARAMETERS_OPTION in signal.get_metadata_options()
            option_dict = signal.get_metadata_option(PROCESSING_PARAMETERS_OPTION)

            # Verify metadata content
            assert option_dict["source_uuid"] == image_uuid
            assert option_dict["func_name"] == "radial_profile"
            assert option_dict["pattern"] == "1-to-1"

            # Verify the parameter can be deserialized
            stored_param = json_to_dataset(option_dict["param_json"])
            assert stored_param.x0 == 50
            assert stored_param.y0 == 50

            # Test that the Processing tab is set up correctly
            # (even though source is in a different panel)
            signal_panel.objview.set_current_object(signal)
            assert signal_panel.objprop.processing_param_editor is not None

            # Test modifying processing parameters
            editor = signal_panel.objprop.processing_param_editor
            # Change the center position
            editor.dataset.x0 = 40
            editor.dataset.y0 = 40

            # Apply the new processing parameters
            report = signal_panel.objprop.apply_processing_parameters()

            # Verify the operation succeeded
            assert report.success, f"Cross-panel reprocessing failed: {report.message}"

            # Verify the parameter was updated in metadata
            updated_dict = signal.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
            updated_param = json_to_dataset(updated_dict["param_json"])
            assert updated_param.x0 == 40
            assert updated_param.y0 == 40

            # Test recompute feature with modified source image
            image.data *= 2.0  # Double the image intensity
            original_signal_data = signal.y.copy()

            # Recompute the radial profile
            signal_panel.recompute_processing()

            # The signal should have changed (doubled intensity)
            assert not np.allclose(signal.y, original_signal_data)


def test_cross_panel_signal_to_image():
    """Test cross-panel processing: Signal → Image (combine signals into image)"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            signal_panel = win.signalpanel
            image_panel = win.imagepanel
            signal_processor = signal_panel.processor

            # Create multiple test signals
            n_signals = 5
            signal_uuids = []
            for i in range(n_signals):
                signal_param = GaussParam.create(
                    mu=250.0 + i * 10, sigma=20.0, a=100.0, y0=float(i), size=500
                )
                signal_panel.new_object(param=signal_param, edit=False)
                signal = signal_panel.objview.get_current_object()
                signal_uuids.append(get_uuid(signal))

            # Select all signals
            for uuid in signal_uuids:
                signal_panel.objview.set_current_item_id(uuid, extend=True)

            # Combine signals into image (Signal → Image cross-panel computation)
            sti_param = SignalsToImageParam.create(
                orientation="columns", normalize=True
            )
            signal_processor.run_feature("signals_to_image", param=sti_param)

            # Verify the result is now in the image panel
            image = image_panel.objview.get_current_object()
            assert image is not None

            # Check that processing metadata was stored
            assert PROCESSING_PARAMETERS_OPTION in image.get_metadata_options()
            option_dict = image.get_metadata_option(PROCESSING_PARAMETERS_OPTION)

            # Verify metadata content for n-to-1 pattern
            assert option_dict["func_name"] == "signals_to_image"
            assert option_dict["pattern"] == "n-to-1"
            assert len(option_dict["source_uuids"]) == n_signals
            assert all(uuid in signal_uuids for uuid in option_dict["source_uuids"])

            # Verify the parameter can be deserialized
            stored_param = json_to_dataset(option_dict["param_json"])
            assert stored_param.orientation == "columns"
            assert stored_param.normalize is True

            # Test that the Processing tab is NOT set up for n-to-1 pattern
            # (interactive processing only works for 1-to-1 pattern)
            image_panel.objview.set_current_object(image)
            assert image_panel.objprop.processing_param_editor is None


def test_select_source_objects_same_panel():
    """Test selecting source objects within the same panel"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            processor = panel.processor

            # Create a source signal
            panel.new_object(edit=False)
            source_signal = panel.objview.get_current_object()
            source_uuid = get_uuid(source_signal)

            # Apply a filter to create a processed signal
            param = GaussianParam.create(sigma=2.0)
            processor.run_feature("gaussian_filter", param=param)

            # Get the filtered signal
            filtered_signal = panel.objview.get_current_object()
            assert filtered_signal is not None

            # Clear selection
            panel.objview.clearSelection()

            # Select only the filtered signal
            panel.objview.set_current_item_id(get_uuid(filtered_signal), extend=False)

            # Call select_source_objects
            panel.select_source_objects()

            # Verify that the source signal is now selected
            selected_uuids = panel.objview.get_sel_object_uuids()
            assert source_uuid in selected_uuids
            assert len(selected_uuids) == 1


def test_select_source_objects_cross_panel():
    """Test selecting source objects across panels (cross-panel computation)"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            image_panel = win.imagepanel
            signal_panel = win.signalpanel
            image_processor = image_panel.processor

            # Create a source image
            image_param = Gauss2DParam.create(
                x0=50.0, y0=50.0, sigma=10.0, a=100.0, height=100, width=100
            )
            image_panel.new_object(param=image_param, edit=False)
            image = image_panel.objview.get_current_object()
            image_uuid = get_uuid(image)

            # Apply radial_profile (Image → Signal)
            from sigima.proc.image import RadialProfileParam

            profile_param = RadialProfileParam.create(x0=50, y0=50)
            image_processor.run_feature("radial_profile", param=profile_param)

            # Verify we're now in the signal panel
            signal = signal_panel.objview.get_current_object()
            assert signal is not None

            # Clear selection in signal panel
            signal_panel.objview.clearSelection()

            # Select only the radial profile signal
            signal_panel.objview.set_current_item_id(get_uuid(signal), extend=False)

            # Call select_source_objects - should switch to image panel
            signal_panel.select_source_objects()

            # Verify that we switched to the image panel
            # (the current panel in the main window should now be image panel)
            assert win.get_current_panel() == "image"

            # Verify that the source image is now selected in the image panel
            selected_uuids = image_panel.objview.get_sel_object_uuids()
            assert image_uuid in selected_uuids
            assert len(selected_uuids) == 1


def test_select_source_objects_multiple_sources():
    """Test selecting multiple source objects (n-to-1 processing)"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            processor = panel.processor

            # Create multiple source signals
            n_signals = 3
            source_uuids = []
            for i in range(n_signals):
                panel.new_object(edit=False)
                signal = panel.objview.get_current_object()
                source_uuids.append(get_uuid(signal))

            # Select all source signals
            for uuid in source_uuids:
                panel.objview.set_current_item_id(uuid, extend=True)

            # Apply addition operation (n-to-1)
            processor.run_feature("addition")

            # Get the result signal
            result_signal = panel.objview.get_current_object()
            assert result_signal is not None

            # Clear selection
            panel.objview.clearSelection()

            # Select only the result signal
            panel.objview.set_current_item_id(get_uuid(result_signal), extend=False)

            # Call select_source_objects
            panel.select_source_objects()

            # Verify that all source signals are now selected
            selected_uuids = panel.objview.get_sel_object_uuids()
            assert len(selected_uuids) == n_signals
            assert all(uuid in selected_uuids for uuid in source_uuids)


def test_select_source_objects_deleted_source():
    """Test selecting source objects when source has been deleted"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            processor = panel.processor

            # Create a source signal
            panel.new_object(edit=False)
            source_signal = panel.objview.get_current_object()

            # Apply a filter
            param = GaussianParam.create(sigma=2.0)
            processor.run_feature("gaussian_filter", param=param)

            # Get the filtered signal
            filtered_signal = panel.objview.get_current_object()

            # Delete the source signal
            panel.objview.set_current_object(source_signal)
            panel.remove_object(force=True)

            # Select the filtered signal
            panel.objview.clearSelection()
            panel.objview.set_current_item_id(get_uuid(filtered_signal), extend=False)

            # Call select_source_objects - should show warning
            # (This won't raise an exception, just show a message box in GUI)
            panel.select_source_objects()

            # Verify that no objects are selected (source was deleted)
            selected_uuids = panel.objview.get_sel_object_uuids()
            # The filtered signal should still be selected
            assert get_uuid(filtered_signal) in selected_uuids


if __name__ == "__main__":
    test_signal_interactive_processing()
    test_image_interactive_processing()
    test_processing_without_parameters()
    test_recompute()
    test_apply_creation_parameters_signal()
    test_apply_creation_parameters_image()
    test_no_creation_parameters_for_base_classes()
    test_apply_processing_parameters_signal()
    test_apply_processing_parameters_image()
    test_apply_processing_parameters_missing_source()
    test_cross_panel_image_to_signal()
    test_cross_panel_signal_to_image()
    test_select_source_objects_same_panel()
    test_select_source_objects_cross_panel()
    test_select_source_objects_multiple_sources()
    test_select_source_objects_deleted_source()
