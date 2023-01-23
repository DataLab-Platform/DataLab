# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Basic application launcher test 4

Create an image object from Scikit-image human mitosis sample,
then open CodraFT to show it.
"""


from skimage.data import human_mitosis

from codraft.core.model.image import create_image
from codraft.tests import codraft_app_context

SHOW = False  # Show test in GUI-based test launcher


def test():
    """Dictionnary/List in metadata (de)serialization test"""
    with codraft_app_context(console=False) as win:
        panel = win.imagepanel
        data = human_mitosis()
        image = create_image("Test image with peaks", data)
        panel.add_object(image)
        panel.processor.compute_peak_detection()
        panel.processor.compute_contour_shape()
        panel.processor.extract_roi()


if __name__ == "__main__":
    test()
