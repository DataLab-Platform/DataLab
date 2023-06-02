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

from qtpy import QtWidgets as QW

from cdl.core.model.base import ANN_KEY
from cdl.env import execenv
from cdl.tests import cdl_app_context
from cdl.tests import data as test_data

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run image tools test scenario"""
    execenv.unattended = True
    with cdl_app_context() as win:
        panel = win.imagepanel
        ima = test_data.create_image_with_annotations()
        panel.add_object(ima)
        panel.open_separate_view().done(QW.QDialog.DialogCode.Accepted)
        orig_metadata = ima.metadata.copy()
        panel.open_separate_view().done(QW.QDialog.DialogCode.Accepted)
        execenv.print("Check [geometric shapes] <--> [plot items] conversion:")
        execenv.print(f"  Comparing {ANN_KEY}: ", end="")
        # open("before.json", mode="wb").write(orig_metadata[ANN_KEY].encode())
        # open("after.json", mode="wb").write(ima.metadata[ANN_KEY].encode())
        assert orig_metadata[ANN_KEY] == ima.metadata[ANN_KEY]
        execenv.print("OK")


if __name__ == "__main__":
    test()
