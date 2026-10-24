# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image offset correction application test

An application test for the image offset correction feature is necessary to ensure
that the feature works correctly with the GUI and the image processing pipeline, as
it involves user interaction and a specific dialog for selecting the offset region.
Experience has shown that this feature can be prone to issues, especially with
dialog interactions, so a dedicated test is essential to catch any regressions or
unexpected behavior in the future.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
import pytest
from sigima.tests.data import get_test_image
from sigima.tests.helpers import rectangle_selection

import datalab.gui.processor.image
from datalab.tests import datalab_test_app_context


def test_offset_correction():
    """Run offset correction application test scenario"""
    with datalab_test_app_context() as win:
        win.imagepanel.add_object(get_test_image("flower.npy"))
        win.imagepanel.processor.compute_offset_correction()


def test_offset_correction_calibrated_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offset correction of an image whose pixels are much smaller than 1"""
    rects = []

    def accept_background_area(dlg) -> bool:
        """Accept the background dialog with its default area"""
        dlg.test_compute_background()
        rects.append(dlg.get_rect_coords())
        return True

    monkeypatch.setattr(
        datalab.gui.processor.image, "exec_dialog", accept_background_area
    )
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        ima = get_test_image("flower.npy")
        ima.data = np.asarray(ima.data, dtype=float)
        ima.set_uniform_coords(0.01, 0.01, -1.0, 2.0)
        panel.add_object(ima)
        panel.processor.compute_offset_correction()
        dst = panel[len(panel)]
        assert dst is not ima
        # The removed offset is the mean of the pixels of the selected area
        x0, y0, x1, y1 = rects[0]
        area = ima.data[rectangle_selection(ima, x0, y0, x1 - x0, y1 - y0)]
        assert area.size > 0
        np.testing.assert_allclose(ima.data - dst.data, area.mean())


if __name__ == "__main__":
    test_offset_correction()
