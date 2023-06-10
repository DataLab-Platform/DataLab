# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Annotations unit test:

  - Create an image with annotations
  - Open dialog (equivalent to click on button "Annotations")
  - Accept dialog without modifying shapes
  - Check if image annotations are still the same
"""

# guitest: show

from guiqwt.builder import make
from qtpy import QtWidgets as QW

from cdl.core.model.base import ANN_KEY
from cdl.env import execenv
from cdl.tests import cdl_app_context
from cdl.tests import data as test_data


def test():
    """Run image tools test scenario"""
    with cdl_app_context() as win:
        panel = win.imagepanel

        # Create image with annotations
        ima1 = test_data.create_multigauss_image()
        ima1.title = "Annotations from items"
        rect = make.annotated_rectangle(100, 100, 200, 200, title="Test")
        circ = make.annotated_circle(300, 300, 400, 400, title="Test")
        elli = make.annotated_ellipse(
            500, 500, 800, 500, 650, 400, 650, 600, title="Test"
        )
        segm = make.annotated_segment(700, 700, 800, 800, title="Test")
        label = make.label("Test", (1000, 1000), (0, 0), "BR")
        ima1.add_annotations_from_items([rect, circ, elli, segm, label])
        ima1.add_annotations_from_file(test_data.get_test_fnames("annotations.json")[0])
        panel.add_object(ima1)

        # Create another image with annotations
        ima2 = test_data.create_annotated_image(title="Annotations from JSON")
        panel.add_object(ima2)

        execenv.print("Check [geometric shapes] <--> [plot items] conversion:")
        execenv.print(f"(comparing {ANN_KEY} metadata)")
        for ima in (ima1, ima2):
            execenv.print(f"  Checking image '{ima.title}': ", end="")
            panel.objview.select_objects([ima])
            # Open separate view
            panel.open_separate_view().done(QW.QDialog.DialogCode.Accepted)
            orig_metadata = ima.metadata.copy()
            panel.open_separate_view().done(QW.QDialog.DialogCode.Accepted)
            # Check if metadata are still the same
            # open("before.json", mode="wb").write(orig_metadata[ANN_KEY].encode())
            # open("after.json", mode="wb").write(ima.metadata[ANN_KEY].encode())
            assert orig_metadata[ANN_KEY] == ima.metadata[ANN_KEY]
            execenv.print("OK")


if __name__ == "__main__":
    test()
