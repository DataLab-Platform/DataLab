# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Example of high-level test scenario

Create an image object from Scikit-image human mitosis sample,
then open DataLab to show it.
"""


from skimage.data import human_mitosis  # pylint: disable=no-name-in-module

from cdl.obj import create_image
from cdl.tests import cdl_app_context

SHOW = False  # Show test in GUI-based test launcher


def test():
    """Dictionnary/List in metadata (de)serialization test"""
    with cdl_app_context(console=False) as win:
        panel = win.imagepanel
        data = human_mitosis()
        image = create_image("Test image with peaks", data)
        panel.add_object(image)
        panel.processor.compute_peak_detection()
        panel.processor.compute_contour_shape()
        panel.processor.extract_roi()


if __name__ == "__main__":
    test()
