# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Save to directory application test for screenshots:

  - Create signals and images
  - Open Save to directory dialog
  - Take screenshots for documentation
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import guidata.config as gcfg
import guidata.dataset as gds

from datalab.env import execenv
from datalab.gui.panel.base import SaveToDirectoryGUIParam
from datalab.tests import datalab_test_app_context, helpers
from datalab.tests.features.common.add_metadata_unit_test import (
    create_test_images,
    create_test_signals,
)

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow


def __save_signals_to_directory(win: DLMainWindow, screenshot: bool = False) -> None:
    """Save signals to directory helper function."""
    execenv.print("Save to directory signal application test:")
    panel = win.signalpanel

    # Add test signals to the panel
    for sig in create_test_signals():
        panel.add_object(sig)

    # Select all signals
    panel.objview.select_objects([1, 2, 3])

    # Get selected objects
    objs = panel.objview.get_sel_objects(include_groups=True)
    extensions = ["csv", "txt", "h5sig"]

    with helpers.WorkdirRestoringTempDir() as tmpdir:
        # Temporarily disable validation to allow creating SaveToDirectoryGUIParam
        # with an empty directory path by default
        old_mode = gcfg.get_validation_mode()
        gcfg.set_validation_mode(gcfg.ValidationMode.DISABLED)
        try:
            # Create and configure the Save to directory dialog
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=gds.DataItemValidationWarning)
                param = SaveToDirectoryGUIParam(objs, extensions)

                # Configure example save parameters
                param.directory = "" if screenshot else tmpdir
                param.basename = "{index:03d}_{title}"
                param.extension = ".csv"
                param.overwrite = False
                param.update_preview()
                execenv.print(f"  Preview: {param.preview}")
                assert "s001: 001_Sine Wave.csv" in param.preview
                execenv.print("  ✓ Save to directory parameter configured correctly")

                # Edit the parameter to show the dialog
                execenv.print("  Opening Save to directory dialog...")
                with execenv.context(screenshot=screenshot):
                    param.edit(
                        parent=win, wordwrap=False, object_name="s_save_to_directory"
                    )

            # Note: We don't actually call panel.save_to_directory() here as we're just
            # testing the dialog display for screenshots
        finally:
            gcfg.set_validation_mode(old_mode)

    execenv.print("==> Signal application test OK")


def __save_images_to_directory(win: DLMainWindow, screenshot: bool = False) -> None:
    """Save images to directory helper function."""
    execenv.print("Save to directory image test:")
    panel = win.imagepanel

    # Add test images to the panel
    for img in create_test_images():
        panel.add_object(img)

    # Select all images
    panel.objview.select_objects([1, 2, 3])

    # Get selected objects
    objs = panel.objview.get_sel_objects(include_groups=True)
    extensions = ["h5ima", "tiff", "png", "jpg"]

    with helpers.WorkdirRestoringTempDir() as tmpdir:
        # Temporarily disable validation to allow creating SaveToDirectoryGUIParam
        # with an empty directory path by default
        old_mode = gcfg.get_validation_mode()
        gcfg.set_validation_mode(gcfg.ValidationMode.DISABLED)
        try:
            # Create and configure the Save to directory dialog
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=gds.DataItemValidationWarning)
                param = SaveToDirectoryGUIParam(objs, extensions)

                # Configure example save parameters
                param.directory = "" if screenshot else tmpdir
                param.basename = "{title}_{index:04d}"
                param.extension = ".png"
                param.overwrite = False
                param.update_preview()
                execenv.print(f"  Preview: {param.preview}")
                assert "i001: Random Noise_0001.png" in param.preview
                execenv.print("  ✓ Save to directory parameter configured correctly")

                # Edit the parameter to show the dialog
                execenv.print("  Opening Save to directory dialog...")
                with execenv.context(screenshot=screenshot):
                    param.edit(
                        parent=win, wordwrap=False, object_name="i_save_to_directory"
                    )

            # Note: We don't actually call panel.save_to_directory() here as we're just
            # testing the dialog display for screenshots
        finally:
            gcfg.set_validation_mode(old_mode)

    execenv.print("==> Image application test OK")


def test_save_to_directory_signals() -> None:
    """Test Save to directory feature for signals."""
    with datalab_test_app_context() as win:
        __save_signals_to_directory(win)


def test_save_to_directory_images() -> None:
    """Test Save to directory feature for images."""
    with datalab_test_app_context() as win:
        __save_images_to_directory(win)


def save_to_directory_screenshots():
    """Generate save to directory screenshots."""
    with execenv.context(unattended=True):
        with datalab_test_app_context() as win:
            execenv.print("Save to directory screenshots test:")
            __save_signals_to_directory(win, screenshot=True)
            __save_images_to_directory(win, screenshot=True)
            execenv.print("==> All screenshot tests completed")


if __name__ == "__main__":
    # test_save_to_directory_signals()
    # test_save_to_directory_images()
    save_to_directory_screenshots()
