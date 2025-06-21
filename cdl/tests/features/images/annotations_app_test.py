# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Annotations application test:

  - Create an image with annotations and ROI
  - Further tests to be done manually: edit "Annotations" and check that
    modifications are taken into account, without affecting the existing ROI
"""

# guitest: show

from cdl.app import run
from cdl.tests import data as test_data
from sigima_.obj import create_image_roi


def test_annotations_app():
    """Annotations test"""
    obj1 = test_data.create_sincos_image()
    obj2 = test_data.create_annotated_image()
    obj2.roi = create_image_roi("rectangle", [10, 10, 60, 400])
    run(console=False, objects=(obj1, obj2), size=(1200, 550))


if __name__ == "__main__":
    test_annotations_app()
