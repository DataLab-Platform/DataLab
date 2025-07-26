# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Image erase application test"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from sigima.tests.data import get_test_image

from datalab.tests import datalab_test_app_context


def test_erase():
    """Run erase application test scenario"""
    with datalab_test_app_context() as win:
        win.imagepanel.add_object(get_test_image("flower.npy"))
        win.imagepanel.processor.compute_erase()


if __name__ == "__main__":
    test_erase()
