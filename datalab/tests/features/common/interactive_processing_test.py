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
from guidata.qthelpers import qt_app_context, qt_wait
from sigima.objects import Gauss2DParam, GaussParam, create_image_roi
from sigima.params import (
    BinningParam,
    ConstantParam,
    GaussianParam,
    MovingAverageParam,
    SignalsToImageParam,
)
from sigima.proc.image import RadialProfileParam

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


def test_no_duplicate_creation_tabs():
    """Test that applying creation parameters multiple times doesn't create
    duplicate tabs.

    This test verifies the fix for the bug where clicking "Apply" in the
    Creation tab would create a new Creation tab instead of reusing the
    existing one. It also verifies that the Creation tab remains current
    after applying changes.
    """
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.imagepanel
            objprop = panel.objprop

            # Create an image with creation parameters
            param = Gauss2DParam.create(x0=50.0, y0=50.0, sigma=10.0, a=100.0)
            panel.new_object(param=param, edit=False)
            image = panel.objview.get_current_object()
            assert image is not None

            # Verify Creation tab was set up
            assert objprop.creation_param_editor is not None
            assert objprop.creation_scroll is not None

            # Count how many Creation tabs exist initially using the widget reference
            initial_index = objprop.tabwidget.indexOf(objprop.creation_scroll)
            assert initial_index >= 0, "Creation tab should be present"

            # Count tabs by checking if they reference the same scroll widget
            initial_count = sum(
                1
                for i in range(objprop.tabwidget.count())
                if objprop.tabwidget.widget(i) is objprop.creation_scroll
            )
            assert initial_count == 1, "Should have exactly one Creation tab initially"

            # Apply creation parameters multiple times
            editor = objprop.creation_param_editor
            for amplitude in [150.0, 200.0, 250.0]:
                editor.dataset.a = amplitude
                objprop.apply_creation_parameters()

                # Wait for the deferred setup_creation_tab to complete
                qt_wait(0.1)

                # Verify that creation_scroll reference still exists
                assert objprop.creation_scroll is not None

                # Count Creation tabs again - should still be just one
                creation_count = sum(
                    1
                    for i in range(objprop.tabwidget.count())
                    if objprop.tabwidget.widget(i) is objprop.creation_scroll
                )
                assert creation_count == 1, (
                    f"Should still have exactly one Creation tab after "
                    f"applying amplitude={amplitude}"
                )

                # Verify that the Creation tab is the current tab
                assert objprop.tabwidget.currentWidget() is objprop.creation_scroll, (
                    f"Creation tab should remain current after "
                    f"applying amplitude={amplitude}"
                )


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


def test_no_duplicate_processing_tabs():
    """Test that applying processing parameters multiple times doesn't create
    duplicate tabs.

    This test verifies the fix for the bug where clicking "Apply" in the
    Processing tab would create a new Processing tab instead of reusing the
    existing one. It also verifies that the Processing tab remains current
    after applying changes.
    """
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.imagepanel
            processor = panel.processor
            objprop = panel.objprop

            # Create a default test image
            panel.new_object(edit=False)
            image = panel.objview.get_current_object()
            assert image is not None

            # Apply addition_constant with initial value
            v0 = 7.0
            processor.run_feature("addition_constant", ConstantParam.create(value=v0))

            # Get the processed image
            processed_ima = panel.objview.get_current_object()
            assert processed_ima is not None

            # Select the processed image to trigger setup_processing_tab
            panel.objview.set_current_object(processed_ima)

            # Verify Processing tab was set up
            assert objprop.processing_param_editor is not None
            assert objprop.processing_scroll is not None

            # Count how many Processing tabs exist initially
            initial_index = objprop.tabwidget.indexOf(objprop.processing_scroll)
            assert initial_index >= 0, "Processing tab should be present"

            # Count tabs by checking if they reference the same scroll widget
            initial_count = sum(
                1
                for i in range(objprop.tabwidget.count())
                if objprop.tabwidget.widget(i) is objprop.processing_scroll
            )
            assert initial_count == 1, (
                "Should have exactly one Processing tab initially"
            )

            # Apply processing parameters multiple times
            editor = objprop.processing_param_editor
            for value in [10.0, 15.0, 20.0]:
                editor.dataset.value = value
                report = objprop.apply_processing_parameters()
                assert report.success

                # Wait for the deferred setup_processing_tab to complete
                qt_wait(0.1)

                # Verify that processing_scroll reference still exists
                assert objprop.processing_scroll is not None

                # Count Processing tabs again - should still be just one
                processing_count = sum(
                    1
                    for i in range(objprop.tabwidget.count())
                    if objprop.tabwidget.widget(i) is objprop.processing_scroll
                )
                assert processing_count == 1, (
                    f"Should still have exactly one Processing tab after "
                    f"applying value={value}"
                )

                # Verify that the Processing tab is the current tab
                assert objprop.tabwidget.currentWidget() is objprop.processing_scroll, (
                    f"Processing tab should remain current after applying value={value}"
                )


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


def test_cross_panel_image_to_signal_group():
    """Test cross-panel processing with groups: Image group → Signal group"""
    with qt_app_context():
        with datalab_test_app_context() as win:
            image_panel = win.imagepanel
            signal_panel = win.signalpanel
            image_processor = image_panel.processor

            # Create a group of test images
            group_name = "Test Images"
            group = image_panel.add_group(group_name)
            group_id = get_uuid(group)

            # Create multiple test images in the group
            n_images = 3
            for i in range(n_images):
                image_param = Gauss2DParam.create(
                    x0=50.0 + i * 5,
                    y0=50.0 + i * 5,
                    sigma=10.0,
                    a=100.0,
                    height=100,
                    width=100,
                )
                image_panel.new_object(param=image_param, edit=False)
                # The new objects are automatically added to the current group

            # Select the entire group
            image_panel.objview.set_current_item_id(group_id, extend=False)

            # Get the initial number of groups in signal panel
            signal_groups_before = len(signal_panel.objmodel.get_groups())

            # Apply radial_profile to the entire group (Image → Signal cross-panel)
            profile_param = RadialProfileParam.create(x0=50, y0=50)
            image_processor.run_feature("radial_profile", param=profile_param)

            # Verify a NEW group was created in the signal panel
            signal_groups_after = len(signal_panel.objmodel.get_groups())
            assert signal_groups_after == signal_groups_before + 1, (
                "Expected a new group to be created for cross-panel computation results"
            )

            # Verify that the new group contains the correct number of signals
            signal_groups = signal_panel.objmodel.get_groups()
            new_group = signal_groups[-1]  # Get the last (newly created) group
            signals_in_group = new_group.get_objects()
            assert len(signals_in_group) == n_images, (
                f"Expected {n_images} signals in new group, got {len(signals_in_group)}"
            )

            # Verify the group name follows the expected pattern
            assert "radial_profile" in new_group.title.lower(), (
                f"Expected group name to contain 'radial_profile', "
                f"got '{new_group.title}'"
            )


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
            for _i in range(n_signals):
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


def test_roi_mask_invalidation_on_size_change():
    """Test that ROI masks are invalidated when image size changes.

    This test verifies the fix for the bug where modifying creation parameters
    that change the image dimensions doesn't invalidate the ROI mask cache,
    resulting in corrupted ROI display.

    Reproduction steps:
    1. Create a 2D gaussian image
    2. Add a rectangular ROI
    3. Change the size of the image (increase width by 50%)
    4. Verify that the ROI mask is properly invalidated and recomputed
    """
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.imagepanel
            objprop = panel.objprop

            # Step 1: Create a 2D gaussian image with specific dimensions
            param = Gauss2DParam.create(
                height=100, width=100, x0=50.0, y0=50.0, sigma=10.0, a=100.0
            )
            panel.new_object(param=param, edit=False)
            image = panel.objview.get_current_object()
            assert image is not None
            assert image.data.shape == (100, 100)

            # Step 2: Add a rectangular ROI
            roi = create_image_roi("rectangle", [20, 20, 40, 40])
            image.roi = roi

            # Verify the ROI mask is created and cached
            mask_before = image.maskdata
            assert mask_before is not None
            assert mask_before.shape == (100, 100)

            # Step 3: Change the image dimensions (increase width by 50%)
            editor = objprop.creation_param_editor
            assert editor is not None
            editor.dataset.width = 150  # Change from 100 to 150

            # Apply the new parameters
            objprop.apply_creation_parameters()

            # Step 4: Verify the image was resized
            updated_image = panel.objview.get_current_object()
            assert get_uuid(updated_image) == get_uuid(image)
            assert updated_image.data.shape == (100, 150)

            # Step 5: Verify the ROI mask was invalidated and will be recomputed
            # with the new dimensions
            mask_after = updated_image.maskdata
            assert mask_after is not None
            assert mask_after.shape == (100, 150), (
                f"ROI mask shape {mask_after.shape} doesn't match "
                f"new image shape {updated_image.data.shape}"
            )

            # The mask should be different from before (different shape)
            assert mask_before.shape != mask_after.shape


def test_roi_mask_invalidation_on_processing_change():
    """Test that ROI masks are invalidated when processing changes image dimensions.

    This test verifies the fix for the bug where reprocessing with parameters
    that change image dimensions doesn't invalidate the ROI mask cache,
    resulting in corrupted ROI display.

    Scenario:
    1. Create a source image
    2. Apply binning (reduces dimensions)
    3. Add ROI to the binned image
    4. Change binning factor (changes dimensions again)
    5. Verify ROI mask is properly recomputed
    """
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.imagepanel
            objprop = panel.objprop

            # Step 1: Create a source image
            param = Gauss2DParam.create(
                height=100, width=100, x0=50.0, y0=50.0, sigma=10.0, a=100.0
            )
            panel.new_object(param=param, edit=False)
            source_image = panel.objview.get_current_object()
            assert source_image is not None

            # Step 2: Apply binning to reduce dimensions
            binning_param = BinningParam.create(sx=2, sy=2)  # 100x100 -> 50x50

            # Use the processor's run_feature method with edit=False
            panel.processor.run_feature("binning", binning_param, edit=False)

            # Get the binned result (last object in the list)
            binned = panel.objview.get_sel_objects()[-1]
            assert binned.data.shape == (50, 50)

            # Step 3: Add a rectangular ROI to the binned image
            roi = create_image_roi("rectangle", [10, 10, 20, 20])
            binned.roi = roi

            # Verify the ROI mask is created and cached
            mask_before = binned.maskdata
            assert mask_before is not None
            assert mask_before.shape == (50, 50)

            # Step 4: Change binning factor via Processing tab
            assert objprop.setup_processing_tab(binned)
            editor = objprop.processing_param_editor
            assert editor is not None

            # Change binning factor from 2x2 to 4x4 (50x50 -> 25x25)
            editor.dataset.sx = 4
            editor.dataset.sy = 4

            # Apply the new processing parameters
            report = objprop.apply_processing_parameters(binned)
            assert report.success

            # Step 5: Verify the image was resized
            assert binned.data.shape == (25, 25)

            # Step 6: Verify the ROI mask was invalidated and will be recomputed
            # with the new dimensions
            mask_after = binned.maskdata
            assert mask_after is not None
            assert mask_after.shape == (25, 25), (
                f"ROI mask shape {mask_after.shape} doesn't match "
                f"new image shape {binned.data.shape}"
            )

            # The mask should be different from before (different shape)
            assert mask_before.shape != mask_after.shape


if __name__ == "__main__":
    test_signal_interactive_processing()
    test_image_interactive_processing()
    test_processing_without_parameters()
    test_recompute()
    test_apply_creation_parameters_signal()
    test_apply_creation_parameters_image()
    test_no_duplicate_creation_tabs()
    test_no_creation_parameters_for_base_classes()
    test_apply_processing_parameters_signal()
    test_apply_processing_parameters_image()
    test_no_duplicate_processing_tabs()
    test_apply_processing_parameters_missing_source()
    test_cross_panel_image_to_signal()
    test_cross_panel_image_to_signal_group()
    test_cross_panel_signal_to_image()
    test_select_source_objects_same_panel()
    test_select_source_objects_cross_panel()
    test_select_source_objects_multiple_sources()
    test_select_source_objects_deleted_source()
    test_roi_mask_invalidation_on_size_change()
    test_roi_mask_invalidation_on_processing_change()
