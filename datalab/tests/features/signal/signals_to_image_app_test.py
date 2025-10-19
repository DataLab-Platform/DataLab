# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Signals to image conversion application test"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import sigima.params
from sigima.enums import NormalizationMethod, SignalsToImageOrientation
from sigima.objects import SignalTypes
from sigima.tests.data import create_periodic_signal

from datalab.env import execenv
from datalab.tests import datalab_test_app_context

SIZE = 100
N_SIGNALS = 10


def test_signals_to_image_app(screenshots: bool = False) -> None:
    """Run signals to image conversion application test scenario

    Args:
        screenshots: If True, take screenshots during the test.
    """
    with datalab_test_app_context(console=False) as win:
        execenv.print("Signals to image conversion application test:")

        # Create multiple test signals
        for i in range(N_SIGNALS):
            sig = create_periodic_signal(
                SignalTypes.SINE, freq=50.0 + i * 10.0, size=SIZE, a=(i + 1) * 0.1
            )
            sig.title = f"Signal {i + 1}"
            win.signalpanel.add_object(sig)

        # Select all signals
        win.signalpanel.objview.select_objects(list(range(1, N_SIGNALS + 1)))

        # Test without normalization, as rows
        p = sigima.params.SignalsToImageParam()
        p.orientation = SignalsToImageOrientation.ROWS
        p.normalize = False
        win.signalpanel.processor.run_feature("signals_to_image", p)

        if screenshots:
            win.statusBar().hide()
            win.take_screenshot("s_signals_to_image_rows")

        # Select all signals again
        win.set_current_panel("signal")
        win.signalpanel.objview.select_objects(list(range(1, N_SIGNALS + 1)))

        # Test without normalization, as columns
        p.orientation = SignalsToImageOrientation.COLUMNS
        win.signalpanel.processor.run_feature("signals_to_image", p)

        if screenshots:
            win.take_screenshot("s_signals_to_image_columns")

        # Select all signals again
        win.set_current_panel("signal")
        win.signalpanel.objview.select_objects(list(range(1, N_SIGNALS + 1)))

        # Test with normalization, as rows
        p.orientation = SignalsToImageOrientation.ROWS
        p.normalize = True
        p.normalize_method = NormalizationMethod.MAXIMUM
        win.signalpanel.processor.run_feature("signals_to_image", p)

        if screenshots:
            win.take_screenshot("s_signals_to_image_normalized")

        # Verify that images were created
        assert len(win.imagepanel) == 3, f"Expected 3 images, got {len(win.imagepanel)}"

        # Switch to image panel to verify dimensions
        win.set_current_panel("image")

        # Verify image dimensions - get objects directly from object model
        images = win.imagepanel.objmodel.get_all_objects()
        assert len(images) == 3, f"Expected 3 images in object model, got {len(images)}"
        img1, img2, _img3 = images
        assert img1.data.shape == (
            N_SIGNALS,
            SIZE,
        ), f"Expected shape ({N_SIGNALS}, {SIZE}), got {img1.data.shape}"
        assert img2.data.shape == (
            SIZE,
            N_SIGNALS,
        ), f"Expected shape ({SIZE}, {N_SIGNALS}), got {img2.data.shape}"

        execenv.print("  ✓ Signals to image conversion test passed")


if __name__ == "__main__":
    test_signals_to_image_app(screenshots=False)
