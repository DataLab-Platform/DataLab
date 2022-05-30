# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Annotations application test:

  - Create an image with annotations and ROI
  - Further tests to be done manually: edit "Annotations" and check that
    modifications are taken into account, without affecting the existing ROI
"""
import numpy as np

from codraft.app import run
from codraft.tests import data as test_data

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Annotations test"""
    obj1 = test_data.create_test_image1()
    obj2 = test_data.create_image_with_annotations()
    obj2.roi = np.array([[10, 10, 60, 400]], int)
    run(console=False, objects=(obj1, obj2), size=(1200, 550))


if __name__ == "__main__":
    test()
