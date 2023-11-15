# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Annotations application test:

  - Create an image with annotations and ROI
  - Further tests to be done manually: edit "Annotations" and check that
    modifications are taken into account, without affecting the existing ROI
"""

# guitest: show

import numpy as np

from cdl.app import run
from cdl.tests import data as test_data


def test():
    """Annotations test"""
    obj1 = test_data.create_sincos_image()
    obj2 = test_data.create_annotated_image()
    obj2.roi = np.array([[10, 10, 60, 400]], int)
    run(console=False, objects=(obj1, obj2), size=(1200, 550))


if __name__ == "__main__":
    test()
