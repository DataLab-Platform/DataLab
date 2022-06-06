# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Annotations unit test:

  - Create an image with annotations
  - Open dialog (equivalent to click on button "Annotations")
  - Accept dialog without modifying shapes
  - Check if image annotations are still the same
"""

from codraft.core.model.base import ANN_KEY
from codraft.tests import codraft_app_context
from codraft.tests import data as test_data
from codraft.utils.env import execenv

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run image tools test scenario"""
    execenv.unattended = True
    with codraft_app_context() as win:
        panel = win.imagepanel
        ima = test_data.create_image_with_annotations()
        panel.add_object(ima)
        panel.open_separate_view()
        orig_metadata = ima.metadata.copy()
        panel.open_separate_view()
        execenv.print("Check [geometric shapes] <--> [plot items] conversion:")
        execenv.print(f"  Comparing {ANN_KEY}: ", end="")
        # open("before.json", mode="wb").write(orig_metadata[ANN_KEY].encode())
        # open("after.json", mode="wb").write(ima.metadata[ANN_KEY].encode())
        assert orig_metadata[ANN_KEY] == ima.metadata[ANN_KEY]
        execenv.print("OK")


if __name__ == "__main__":
    test()
