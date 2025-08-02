# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
New signal/image test

Testing GUI functions related to signal/image creation.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import sigima.objects
from guidata.qthelpers import qt_app_context
from sigima.tests.vistools import view_curves, view_images

from datalab.env import execenv
from datalab.gui.newobject import create_image_gui, create_signal_gui


def test_new_signal() -> None:
    """Test new signal feature"""
    edit = not execenv.unattended
    with qt_app_context():
        signal = create_signal_gui(None, edit=edit)
        if signal is not None:
            data = (signal.x, signal.y)
            view_curves([data], name=test_new_signal.__name__, title=signal.title)


def test_new_image() -> None:
    """Test new image feature"""
    # Test with no input parameter
    edit = not execenv.unattended
    with qt_app_context():
        image = create_image_gui(None, edit=edit)
        if image is not None:
            view_images(image.data, name=test_new_image.__name__, title=image.title)
        # Test with parametered 2D-Gaussian
        param = sigima.objects.Gauss2DParam.create(x0=3, y0=3, sigma=5)
        image = create_image_gui(param, edit=edit)
        if image is not None:
            view_images(image.data, name=test_new_image.__name__, title=image.title)


if __name__ == "__main__":
    test_new_signal()
    test_new_image()
