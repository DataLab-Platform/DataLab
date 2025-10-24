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
import numpy as np
import sigima.objects

from datalab.env import execenv
from datalab.gui.panel.base import AddMetadataParam
from datalab.tests import datalab_test_app_context

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow


def __add_metadata_to_signals(win: DLMainWindow, screenshot: bool = False) -> None:
    """Add metadata to signals helper function."""
    execenv.print("Add metadata signal application test:")
    panel = win.signalpanel

    # Create some test signals with different characteristics
    x = np.linspace(0, 10, 100)

    # Signal 1: Sine wave
    y1 = np.sin(x)
    sig1 = sigima.objects.create_signal(
        title="Sine Wave",
        x=x,
        y=y1,
        metadata={"frequency": "1 Hz", "amplitude": "1.0"},
    )
    panel.add_object(sig1)

    # Signal 2: Cosine wave
    y2 = np.cos(x * 2)
    sig2 = sigima.objects.create_signal(
        title="Cosine Wave",
        x=x,
        y=y2,
        metadata={"frequency": "2 Hz", "amplitude": "1.0"},
    )
    panel.add_object(sig2)

    # Signal 3: Exponential decay
    y3 = np.exp(-x / 3)
    sig3 = sigima.objects.create_signal(
        title="Exponential Decay",
        x=x,
        y=y3,
        metadata={"time_constant": "3 s"},
    )
    panel.add_object(sig3)

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

    # Create some test images with different characteristics

    # Image 1: Random noise
    data1 = np.random.rand(100, 100)
    img1 = sigima.objects.create_image(
        title="Random Noise",
        data=data1,
        metadata={"type": "noise", "source": "random"},
    )
    panel.add_object(img1)

    # Image 2: Gaussian pattern
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    data2 = np.exp(-(X**2 + Y**2))
    img2 = sigima.objects.create_image(
        title="Gaussian Pattern",
        data=data2,
        metadata={"type": "gaussian", "sigma": "1.0"},
    )
    panel.add_object(img2)

    # Image 3: Checkerboard
    data3 = np.zeros((100, 100))
    data3[::10, ::10] = 1
    data3[5::10, 5::10] = 1
    img3 = sigima.objects.create_image(
        title="Checkerboard",
        data=data3,
        metadata={"type": "pattern", "period": "10 px"},
    )
    panel.add_object(img3)

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
