# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test UUID preservation when reopening workspace

This test verifies that when a workspace is saved to HDF5, cleared, and reopened,
the object UUIDs are preserved so that processing parameter references (source_uuid,
source_uuids) remain valid. This ensures features like "Show source" and "Recompute"
continue to work after reopening a workspace.

Related issue: Processing tab features don't work after reopening workspace because
UUIDs are regenerated.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import os.path as osp

from guidata.qthelpers import qt_app_context
from sigima.params import GaussianParam

from datalab.gui.processor.base import PROCESSING_PARAMETERS_OPTION
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context, helpers


def test_uuid_preservation_signals():
    """Test that signal UUIDs are preserved when reopening workspace"""
    with qt_app_context():
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            with datalab_test_app_context(console=False) as win:
                panel = win.signalpanel
                processor = panel.processor

                # Step 1: Create a test signal
                panel.new_object()
                signal = panel.objview.get_current_object()
                assert signal is not None
                original_signal_uuid = get_uuid(signal)

                # Step 2: Apply a processing operation with parameters
                param = GaussianParam.create(sigma=2.0)
                processor.compute_1_to_1(
                    processor.get_feature("gaussian_filter").function,
                    param=param,
                    title="Gaussian filter",
                )

                # Step 3: Get the filtered signal and verify metadata
                filtered_sig = panel.objview.get_current_object()
                assert filtered_sig is not None
                original_filtered_uuid = get_uuid(filtered_sig)

                # Verify processing metadata was stored correctly
                assert (
                    PROCESSING_PARAMETERS_OPTION in filtered_sig.get_metadata_options()
                )
                option_dict = filtered_sig.get_metadata_option(
                    PROCESSING_PARAMETERS_OPTION
                )
                assert option_dict["source_uuid"] == original_signal_uuid
                assert option_dict["func_name"] == "gaussian_filter"

                # Step 4: Save workspace to HDF5
                fname = osp.join(tmpdir, "test_uuid_preservation.h5")
                win.save_to_h5_file(fname)

                # Step 5: Remove all objects (clear workspace)
                panel.remove_all_objects()
                assert len(panel.objmodel) == 0

                # Step 6: Reopen workspace
                win.open_h5_files([fname], import_all=True, reset_all=True)

                # Step 7: Verify UUIDs are preserved
                signal_after = panel.objmodel.get_object_from_number(1)
                filtered_sig_after = panel.objmodel.get_object_from_number(2)

                signal_uuid_after = get_uuid(signal_after)
                filtered_uuid_after = get_uuid(filtered_sig_after)

                # CRITICAL: UUIDs must be preserved after reload
                assert signal_uuid_after == original_signal_uuid, (
                    f"Signal UUID changed after reload: "
                    f"{original_signal_uuid} -> {signal_uuid_after}"
                )
                assert filtered_uuid_after == original_filtered_uuid, (
                    f"Filtered signal UUID changed after reload: "
                    f"{original_filtered_uuid} -> {filtered_uuid_after}"
                )

                # Step 8: Verify processing parameters still reference correct source
                option_dict_after = filtered_sig_after.get_metadata_option(
                    PROCESSING_PARAMETERS_OPTION
                )
                assert option_dict_after["source_uuid"] == signal_uuid_after, (
                    f"Processing parameter source_uuid doesn't match: "
                    f"{option_dict_after['source_uuid']} != {signal_uuid_after}"
                )

                # Step 9: Verify "Show source" feature would work
                # (by checking that the source object can be found)
                source_obj = win.find_object_by_uuid(option_dict_after["source_uuid"])
                assert source_obj is signal_after, (
                    "Cannot find source object by UUID - "
                    "'Show source' feature would fail"
                )


def test_uuid_preservation_images():
    """Test that image UUIDs are preserved when reopening workspace"""
    with qt_app_context():
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            with datalab_test_app_context(console=False) as win:
                panel = win.imagepanel
                processor = panel.processor

                # Step 1: Create a test image
                panel.new_object()
                image = panel.objview.get_current_object()
                assert image is not None
                original_image_uuid = get_uuid(image)

                # Step 2: Apply a processing operation
                param = GaussianParam.create(sigma=1.5)
                processor.compute_1_to_1(
                    processor.get_feature("gaussian_filter").function,
                    param=param,
                    title="Gaussian filter",
                )

                # Step 3: Get the filtered image and verify metadata
                filtered_img = panel.objview.get_current_object()
                assert filtered_img is not None
                original_filtered_uuid = get_uuid(filtered_img)

                # Verify processing metadata
                option_dict = filtered_img.get_metadata_option(
                    PROCESSING_PARAMETERS_OPTION
                )
                assert option_dict["source_uuid"] == original_image_uuid

                # Step 4: Save, clear, and reopen workspace
                fname = osp.join(tmpdir, "test_uuid_preservation_images.h5")
                win.save_to_h5_file(fname)
                panel.remove_all_objects()
                win.open_h5_files([fname], import_all=True, reset_all=True)

                # Step 5: Verify UUIDs are preserved
                image_after = panel.objmodel.get_object_from_number(1)
                filtered_img_after = panel.objmodel.get_object_from_number(2)

                assert get_uuid(image_after) == original_image_uuid
                assert get_uuid(filtered_img_after) == original_filtered_uuid

                # Step 6: Verify processing parameters still work
                option_dict_after = filtered_img_after.get_metadata_option(
                    PROCESSING_PARAMETERS_OPTION
                )
                assert option_dict_after["source_uuid"] == get_uuid(image_after)
                source_obj = win.find_object_by_uuid(option_dict_after["source_uuid"])
                assert source_obj is image_after


def test_uuid_regeneration_on_import():
    """Test that UUIDs are regenerated when importing (not resetting workspace)"""
    with qt_app_context():
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            with datalab_test_app_context(console=False) as win:
                panel = win.signalpanel

                # Step 1: Create and save a signal
                panel.new_object()
                signal1 = panel.objview.get_current_object()
                uuid1 = get_uuid(signal1)

                fname = osp.join(tmpdir, "test_import.h5")
                win.save_to_h5_file(fname)

                # Step 2: Import the same file (reset_all=False, import_all=True)
                # This should regenerate UUIDs to avoid conflicts
                win.open_h5_files([fname], import_all=True, reset_all=False)

                # Step 3: Verify we now have 2 signals with different UUIDs
                assert len(panel.objmodel) == 2

                signal1_after = panel.objmodel.get_object_from_number(1)
                signal2_imported = panel.objmodel.get_object_from_number(2)

                uuid1_after = get_uuid(signal1_after)
                uuid2 = get_uuid(signal2_imported)

                # Original signal should keep its UUID
                assert uuid1_after == uuid1

                # Imported signal should have a NEW UUID (different from original)
                assert uuid2 != uuid1, (
                    "Imported signal should have a new UUID, "
                    f"but got same as original: {uuid2} == {uuid1}"
                )

                # Step 4: Import the same file again
                win.open_h5_files([fname], import_all=True, reset_all=False)

                # Step 5: Verify we now have 3 signals, all with different UUIDs
                assert len(panel.objmodel) == 3

                signal3_imported = panel.objmodel.get_object_from_number(3)
                uuid3 = get_uuid(signal3_imported)

                # Third signal should have yet another different UUID
                assert uuid3 != uuid1
                assert uuid3 != uuid2
                assert len({uuid1, uuid2, uuid3}) == 3, (
                    "All three signals should have unique UUIDs"
                )


def test_uuid_preservation_empty_workspace():
    """Test that UUIDs are preserved when opening HDF5 in empty workspace"""
    with qt_app_context():
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            # Create and save a workspace with processing history
            with datalab_test_app_context(console=False) as win1:
                panel = win1.signalpanel
                processor = panel.processor

                # Create signal and apply processing
                panel.new_object()
                signal = panel.objview.get_current_object()
                original_uuid = get_uuid(signal)

                param = GaussianParam.create(sigma=2.0)
                processor.compute_1_to_1(
                    processor.get_feature("gaussian_filter").function,
                    param=param,
                    title="Gaussian filter",
                )

                filtered_sig = panel.objview.get_current_object()
                original_filtered_uuid = get_uuid(filtered_sig)

                # Save workspace
                fname = osp.join(tmpdir, "test_empty_workspace.h5")
                win1.save_to_h5_file(fname)

            # Open the file in a NEW empty workspace (simulating startup)
            with datalab_test_app_context(console=False) as win2:
                panel = win2.signalpanel

                # Workspace is empty - UUIDs should be preserved (reset_all=True)
                # Note: reset_all is None, so it should auto-detect empty workspace
                win2.open_h5_files([fname], import_all=True, reset_all=None)

                # Verify UUIDs are preserved
                signal_loaded = panel.objmodel.get_object_from_number(1)
                filtered_loaded = panel.objmodel.get_object_from_number(2)

                assert get_uuid(signal_loaded) == original_uuid, (
                    "Signal UUID should be preserved when opening in empty workspace"
                )
                assert get_uuid(filtered_loaded) == original_filtered_uuid, (
                    "Filtered signal UUID should be preserved "
                    "when opening in empty workspace"
                )

                # Verify processing parameters still work
                option_dict = filtered_loaded.get_metadata_option(
                    PROCESSING_PARAMETERS_OPTION
                )
                assert option_dict["source_uuid"] == get_uuid(signal_loaded)
                source_obj = win2.find_object_by_uuid(option_dict["source_uuid"])
                assert source_obj is signal_loaded


if __name__ == "__main__":
    test_uuid_preservation_signals()
    test_uuid_preservation_images()
    test_uuid_regeneration_on_import()
    test_uuid_preservation_empty_workspace()
    print("All UUID preservation tests passed!")
