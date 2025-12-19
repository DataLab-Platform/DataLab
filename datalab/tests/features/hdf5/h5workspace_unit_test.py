# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
HDF5 workspace API unit tests
-----------------------------

Tests for the headless HDF5 workspace API methods:
- load_h5_workspace: Load native DataLab HDF5 files without GUI elements
- save_h5_workspace: Save workspace to native DataLab HDF5 file without GUI elements

These methods are designed for use from the internal console where Qt GUI elements
would cause thread-safety issues.
"""

# guitest: show

import os.path as osp

import h5py
import pytest
from sigima.tests.data import create_noisy_gaussian_image, create_paracetamol_signal

from datalab.tests import datalab_test_app_context, helpers


def test_save_and_load_h5_workspace():
    """Test save_h5_workspace and load_h5_workspace methods"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            # === Create test objects
            sig1 = create_paracetamol_signal()
            win.signalpanel.add_object(sig1)

            ima1 = create_noisy_gaussian_image()
            win.imagepanel.add_object(ima1)

            # Store object counts and titles for verification
            sig_count_before = len(win.signalpanel.objmodel)
            ima_count_before = len(win.imagepanel.objmodel)
            sig_title = sig1.title
            ima_title = ima1.title

            # === Test save_h5_workspace
            fname = osp.join(tmpdir, "test_workspace.h5")
            win.save_h5_workspace(fname)
            assert osp.exists(fname), "HDF5 file was not created"

            # === Clear workspace
            for panel in win.panels:
                panel.remove_all_objects()

            assert len(win.signalpanel.objmodel) == 0
            assert len(win.imagepanel.objmodel) == 0

            # === Test load_h5_workspace
            win.load_h5_workspace([fname], reset_all=True)

            # Verify objects were restored
            assert len(win.signalpanel.objmodel) == sig_count_before
            assert len(win.imagepanel.objmodel) == ima_count_before

            # Verify titles (get objects in order from groups)
            loaded_sig = win.signalpanel.objmodel.get_all_objects()[0]
            loaded_ima = win.imagepanel.objmodel.get_all_objects()[0]
            assert loaded_sig.title == sig_title
            assert loaded_ima.title == ima_title


def test_load_h5_workspace_invalid_file():
    """Test load_h5_workspace raises ValueError for non-native HDF5 files"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            # Create a non-native HDF5 file (just an empty HDF5)
            fname = osp.join(tmpdir, "not_native.h5")
            with h5py.File(fname, "w") as f:
                f.create_dataset("dummy", data=[1, 2, 3])

            # Should raise ValueError for non-native file
            with pytest.raises(ValueError, match="not a native DataLab HDF5 file"):
                win.load_h5_workspace([fname])


def test_load_h5_workspace_append():
    """Test load_h5_workspace with reset_all=False (append mode)"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            # Create and save first signal
            sig1 = create_paracetamol_signal()
            sig1.title = "Signal 1"
            win.signalpanel.add_object(sig1)

            fname = osp.join(tmpdir, "workspace1.h5")
            win.save_h5_workspace(fname)

            # Clear and create second signal
            win.signalpanel.remove_all_objects()
            sig2 = create_paracetamol_signal()
            sig2.title = "Signal 2"
            win.signalpanel.add_object(sig2)

            assert len(win.signalpanel.objmodel) == 1

            # Load first workspace with reset_all=False (append)
            win.load_h5_workspace([fname], reset_all=False)

            # Should now have both signals (appended)
            assert len(win.signalpanel.objmodel) == 2


def test_save_h5_workspace_modified_flag():
    """Test that save_h5_workspace clears the modified flag"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            # Create an object (this sets modified flag)
            sig1 = create_paracetamol_signal()
            win.signalpanel.add_object(sig1)

            # Workspace should be modified
            assert win.is_modified()

            # Save workspace
            fname = osp.join(tmpdir, "test.h5")
            win.save_h5_workspace(fname)

            # Modified flag should be cleared
            assert not win.is_modified()


if __name__ == "__main__":
    test_save_and_load_h5_workspace()
    test_load_h5_workspace_invalid_file()
    test_load_h5_workspace_append()
    test_save_h5_workspace_modified_flag()
