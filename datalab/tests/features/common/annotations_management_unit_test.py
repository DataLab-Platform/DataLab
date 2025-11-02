# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for annotation copy/paste functionality."""

import os.path as osp

from sigima.tests import data as test_data

from datalab.env import execenv
from datalab.tests import datalab_test_app_context, helpers


def test_annotations_copy_paste():
    """Test copying and pasting annotations between objects."""
    with execenv.context(unattended=True):
        with datalab_test_app_context() as win:
            panel = win.signalpanel

            # Create two signals
            sig1 = test_data.create_paracetamol_signal()
            sig2 = test_data.create_paracetamol_signal()

            # Add annotations to first signal
            orig_annotations = [
                {"type": "label", "text": "Peak 1"},
                {"type": "label", "text": "Peak 2"},
            ]
            sig1.set_annotations(orig_annotations)

            # Add objects to panel - sig1 will be selected after this
            panel.add_object(sig1)

            # Copy from first signal (which is currently selected)
            panel.copy_annotations()

            # Add second signal - sig2 will be selected after this
            panel.add_object(sig2)

            # Paste to second signal (which is now selected)
            panel.paste_annotations()

            # Verify annotations were copied
            assert sig2.get_annotations() == orig_annotations


def test_annotations_import_export():
    """Test importing and exporting annotations to file."""
    with execenv.context(unattended=True):
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            fname = osp.join(tmpdir, "test.dlabann")
            with datalab_test_app_context() as win:
                panel = win.signalpanel

                # Create signal with annotations
                sig = test_data.create_paracetamol_signal()
                orig_annotations = [{"type": "label", "text": "Test annotation"}]
                sig.set_annotations(orig_annotations)

                panel.add_object(sig)

                # Export annotations
                panel.export_annotations_from_file(fname)

                # Clear annotations to test import
                sig.clear_annotations()
                assert len(sig.get_annotations()) == 0

                # Import annotations
                panel.import_annotations_from_file(fname)

                # Verify annotations were imported
                assert sig.get_annotations() == orig_annotations


def test_annotations_delete():
    """Test deleting annotations from objects."""
    with execenv.context(unattended=True):
        with datalab_test_app_context() as win:
            panel = win.signalpanel

            # Create signal with annotations
            sig = test_data.create_paracetamol_signal()
            sig.set_annotations(
                [
                    {"type": "label", "text": "To be deleted"},
                ]
            )

            panel.add_object(sig)

            # Verify annotations exist
            assert len(sig.get_annotations()) == 1

            # Delete annotations
            panel.delete_annotations()

            # Verify annotations were deleted
            assert len(sig.get_annotations()) == 0


def test_annotations_action_states():
    """Test that action states are updated after annotation operations.

    This test verifies the fix for the issue where action enable states
    were not updated after operations like delete_annotations, requiring
    the user to select another object for the actions to update.

    Note: We can't easily test the UI action states in automated tests,
    but we verify that the underlying state update mechanism is called.
    """
    with execenv.context(unattended=True):
        with datalab_test_app_context() as win:
            panel = win.signalpanel

            # Create signal with annotations
            sig = test_data.create_paracetamol_signal()
            orig_annotations = [{"type": "label", "text": "Test"}]
            sig.set_annotations(orig_annotations)

            panel.add_object(sig)

            # Verify object has annotations
            assert len(sig.get_annotations()) == 1

            # Delete annotations - this should update action states
            panel.delete_annotations()

            # Verify annotations were deleted
            assert len(sig.get_annotations()) == 0

            # Verify clipboard is empty
            assert len(panel.annotations_clipboard) == 0

            # Re-add annotations and copy - this should update action states
            sig.set_annotations(orig_annotations)
            panel.selection_changed()  # Refresh states
            panel.copy_annotations()

            # Verify clipboard now has annotations
            assert len(panel.annotations_clipboard) == 1

            # Clear annotations, paste, and verify
            sig.clear_annotations()
            panel.paste_annotations()
            assert len(sig.get_annotations()) == 1


if __name__ == "__main__":
    test_annotations_copy_paste()
    test_annotations_import_export()
    test_annotations_delete()
    test_annotations_action_states()
