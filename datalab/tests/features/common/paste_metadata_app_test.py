# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Paste metadata application test for screenshots:

  - Create signals and images with metadata
  - Copy metadata to clipboard
  - Open Paste metadata dialog
  - Take screenshots for documentation
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from typing import TYPE_CHECKING

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.base import PasteMetadataParam
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.add_metadata_unit_test import (
    create_test_images,
    create_test_signals,
)

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow


def __paste_metadata_for_signals(win: DLMainWindow, screenshot: bool = False) -> None:
    """Paste metadata for signals helper function."""
    execenv.print("Paste metadata signal application test:")
    panel = win.signalpanel

    # Add test signals to the panel
    for sig in create_test_signals():
        panel.add_object(sig)

    # Select first signal and copy its metadata to clipboard
    panel.objview.select_objects([1])
    panel.copy_metadata()
    execenv.print("  ✓ Metadata copied to clipboard")

    # Select remaining signals as paste targets
    panel.objview.select_objects([2, 3])

    # Create the Paste metadata dialog (same as production code in paste_metadata())
    param = PasteMetadataParam(
        _("Paste metadata"),
        comment=_(
            "Select what to keep from the clipboard.<br><br>"
            "Result shapes and annotations, if kept, will be merged with "
            "existing ones. <u>All other metadata will be replaced</u>."
        ),
    )

    # Edit the parameter to show the dialog
    execenv.print("  Opening Paste metadata dialog...")
    with execenv.context(screenshot=screenshot):
        param.edit(parent=win, object_name="s_paste_metadata")

    execenv.print("==> Signal application test OK")


def __paste_metadata_for_images(win: DLMainWindow, screenshot: bool = False) -> None:
    """Paste metadata for images helper function."""
    execenv.print("Paste metadata image application test:")
    panel = win.imagepanel

    # Add test images to the panel
    for img in create_test_images():
        panel.add_object(img)

    # Select first image and copy its metadata to clipboard
    panel.objview.select_objects([1])
    panel.copy_metadata()
    execenv.print("  ✓ Metadata copied to clipboard")

    # Select remaining images as paste targets
    panel.objview.select_objects([2, 3])

    # Create the Paste metadata dialog (same as production code in paste_metadata())
    param = PasteMetadataParam(
        _("Paste metadata"),
        comment=_(
            "Select what to keep from the clipboard.<br><br>"
            "Result shapes and annotations, if kept, will be merged with "
            "existing ones. <u>All other metadata will be replaced</u>."
        ),
    )

    # Edit the parameter to show the dialog
    execenv.print("  Opening Paste metadata dialog...")
    with execenv.context(screenshot=screenshot):
        param.edit(parent=win, object_name="i_paste_metadata")

    execenv.print("==> Image application test OK")


def test_paste_metadata_for_signals() -> None:
    """Test Paste metadata feature for signals."""
    with datalab_test_app_context() as win:
        __paste_metadata_for_signals(win)


def test_paste_metadata_for_images() -> None:
    """Test Paste metadata feature for images."""
    with datalab_test_app_context() as win:
        __paste_metadata_for_images(win)


def paste_metadata_screenshots():
    """Generate paste metadata screenshots."""
    with execenv.context(unattended=True):
        with datalab_test_app_context() as win:
            execenv.print("Paste metadata screenshots test:")
            __paste_metadata_for_signals(win, screenshot=True)
            __paste_metadata_for_images(win, screenshot=True)
            execenv.print("==> All screenshot tests completed")


if __name__ == "__main__":
    paste_metadata_screenshots()
