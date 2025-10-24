# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Add metadata application test for screenshots:

  - Create signals and images
  - Open Add metadata dialog
  - Take screenshots for documentation
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import guidata.dataset as gds

from datalab.env import execenv
from datalab.gui.panel.base import AddMetadataParam
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.add_metadata_unit_test import (
    create_test_images,
    create_test_signals,
)

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow


def __add_metadata_to_signals(win: DLMainWindow, screenshot: bool = False) -> None:
    """Add metadata to signals helper function."""
    execenv.print("Add metadata signal application test:")
    panel = win.signalpanel

    # Add test signals to the panel
    for sig in create_test_signals():
        panel.add_object(sig)

    # Select all signals
    panel.objview.select_objects([1, 2, 3])

    # Create and configure the Add metadata dialog
    objs = panel.objview.get_sel_objects(include_groups=True)
    param = AddMetadataParam(objs)

    # Configure example metadata
    param.metadata_key = "experiment_id"
    param.value_pattern = "EXP_{index:03d}"
    param.conversion = "string"
    param.update_preview()
    execenv.print(f"  Preview: {param.preview}")
    assert "experiment_id" in param.preview
    execenv.print("  ✓ Add metadata parameter configured correctly")

    # Edit the parameter to show the dialog
    execenv.print("  Opening Add metadata dialog...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=gds.DataItemValidationWarning)
        with execenv.context(screenshot=screenshot):
            param.edit(parent=win, wordwrap=False, object_name="s_add_metadata")

    # Run DataLab's metadata addition functionality:
    panel.add_metadata(param)

    execenv.print("==> Signal application test OK")


def __add_metadata_to_images(win: DLMainWindow, screenshot: bool = False) -> None:
    """Add metadata to images helper function."""
    execenv.print("Add metadata image test:")
    panel = win.imagepanel

    # Add test images to the panel
    for img in create_test_images():
        panel.add_object(img)

    # Select all images
    panel.objview.select_objects([1, 2, 3])

    # Create and configure the Add metadata dialog
    objs = panel.objview.get_sel_objects(include_groups=True)
    param = AddMetadataParam(objs)

    # Configure example metadata
    param.metadata_key = "sample_id"
    param.value_pattern = "SAMPLE_{index:04d}_{title:upper}"
    param.conversion = "string"
    param.update_preview()
    execenv.print(f"  Preview: {param.preview}")
    assert "sample_id" in param.preview
    execenv.print("  ✓ Add metadata parameter configured correctly")

    # Edit the parameter to show the dialog
    execenv.print("  Opening Add metadata dialog...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=gds.DataItemValidationWarning)
        with execenv.context(screenshot=screenshot):
            param.edit(parent=win, wordwrap=False, object_name="i_add_metadata")

    # Run DataLab's metadata addition functionality:
    panel.add_metadata(param)

    execenv.print("==> Image application test OK")


def test_add_metadata_to_signals() -> None:
    """Test Add metadata feature for signals."""
    with datalab_test_app_context() as win:
        __add_metadata_to_signals(win)


def test_add_metadata_to_images() -> None:
    """Test Add metadata feature for images."""
    with datalab_test_app_context() as win:
        __add_metadata_to_images(win)


def add_metadata_screenshots():
    """Generate add metadata screenshots."""
    with execenv.context(unattended=True):
        with datalab_test_app_context() as win:
            execenv.print("Add metadata screenshots test:")
            __add_metadata_to_signals(win, screenshot=True)
            __add_metadata_to_images(win, screenshot=True)
            execenv.print("==> All screenshot tests completed")


if __name__ == "__main__":
    # test_add_metadata_to_signals()
    # test_add_metadata_to_images()
    add_metadata_screenshots()
