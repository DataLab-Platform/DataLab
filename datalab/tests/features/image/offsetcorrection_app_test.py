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

from sigima.tests.data import get_test_image

from datalab.tests import datalab_test_app_context


def test_offset_correction():
    """Run offset correction application test scenario"""
    with datalab_test_app_context() as win:
        win.imagepanel.add_object(get_test_image("flower.npy"))
        win.imagepanel.processor.compute_offset_correction()


if __name__ == "__main__":
    test_offset_correction()
