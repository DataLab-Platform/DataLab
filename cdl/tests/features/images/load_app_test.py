# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Load application test: high number of images
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
import pytest

import cdl.obj
import sigima.param as sp
from cdl.env import execenv
from cdl.tests import cdltest_app_context


def create_random_test_data(size: tuple[int, int] | None = None) -> cdl.obj.ImageObj:
    """Create a test image, based on a fast algorithm, to be able to generate
    a high number of images.

    Args:
        size: Size of the image.
    """
    if size is None:
        size = (2048, 2048)
    # Create a base image with low frequency shapes:
    x = np.linspace(-1, 1, size[0])
    y = np.linspace(-1, 1, size[1])
    xx, yy = np.meshgrid(x, y)
    a1, a2, f1, f2 = np.random.rand(4)
    data = a1 * np.sin(2 * np.pi * xx * (1 + f1 * 0.2)) + a2 * np.cos(
        2 * np.pi * yy * (1 + f2 * 0.2)
    )
    # Add some random noise:
    data += 0.1 * np.random.randn(*data.shape)
    image = cdl.obj.create_image("Random test image", data)
    return image


@pytest.mark.skip("This a load test, not a functional test")
def test_high_number_of_images() -> None:
    """Run a test with a high number of images."""
    nb = 30
    execenv.print("Creating images", end="")
    images = []
    for idx in range(nb):
        execenv.print(".", end="")
        ima = create_random_test_data()
        ima.title += f" {idx}"
        images.append(ima)
    execenv.print(" done")
    with cdltest_app_context() as win:
        panel = win.imagepanel
        for ima in images:
            panel.add_object(ima)
        panel.objview.select_groups()

        # Comment the two following lines to check if DataLab only shows the last image
        # when they are all superposed
        param = sp.GridParam.create(cols=10)
        panel.processor.distribute_on_grid(param)

        panel.duplicate_object()


if __name__ == "__main__":
    test_high_number_of_images()
