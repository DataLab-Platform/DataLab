# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
View images side-by-side test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
import sigima.objects

from datalab.tests import datalab_test_app_context


def test_view_images_side_by_side() -> None:
    """Test viewing multiple images side-by-side."""
    with datalab_test_app_context() as win:
        panel = win.imagepanel

        # Create several test images with different content
        for i in range(5):
            x = np.linspace(-5, 5, 100)
            y = np.linspace(-5, 5, 100)
            xx, yy = np.meshgrid(x, y)

            # Different patterns for each image
            if i == 0:
                data = np.exp(-(xx**2 + yy**2) / (2 * (i + 1)))
            elif i == 1:
                data = np.sin(xx) * np.cos(yy)
            elif i == 2:
                data = np.abs(xx) + np.abs(yy)
            elif i == 3:
                data = xx**2 - yy**2
            else:
                data = np.random.randn(100, 100)

            image = sigima.objects.create_image(f"Test Image {i + 1}", data)
            panel.add_object(image)

        # Select all images (1 to 5)
        panel.objview.select_objects(list(range(1, 6)))

        # Open side-by-side view
        panel.view_images_side_by_side()


if __name__ == "__main__":
    test_view_images_side_by_side()
