# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Metadata import/export unit test:

  - Create an image with annotations and result shapes
  - Add the image to DataLab
  - Export image metadata to file (JSON)
  - Delete image metadata
  - Import image metadata from previous file
  - Check if image metadata is the same as the original image
"""

import os.path as osp

import numpy as np

from cdl.env import execenv
from cdl.tests import cdl_app_context
from cdl.tests import data as test_data
from cdl.utils import tests

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run image tools test scenario"""
    execenv.unattended = True
    with tests.temporary_directory() as tmpdir:
        fname = osp.join(tmpdir, "test.json")
        with cdl_app_context() as win:
            panel = win.imagepanel
            ima = test_data.create_image_with_annotations()
            for mshape in test_data.create_resultshapes():
                mshape.add_to(ima)
            panel.add_object(ima)
            orig_metadata = ima.metadata.copy()
            panel.export_metadata_from_file(fname)
            panel.delete_metadata()
            assert len(ima.metadata) == 0
            panel.import_metadata_from_file(fname)
            execenv.print("Check metadata export <--> import features:")
            for key, value in orig_metadata.items():
                execenv.print(f"  Checking {key} key value...", end="")
                if isinstance(value, np.ndarray):
                    assert (value == ima.metadata[key]).all()
                else:
                    assert value == ima.metadata[key]
                execenv.print("OK")


if __name__ == "__main__":
    test()
