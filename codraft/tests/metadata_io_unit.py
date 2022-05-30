# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Metadata import/export unit test:

  - Create an image with annotations and result shapes
  - Add the image to CodraFT
  - Export image metadata to file (JSON)
  - Delete image metadata
  - Import image metadata from previous file
  - Check if image metadata is the same as the original image
"""

import os.path as osp

import numpy as np

from codraft.tests import codraft_app_context
from codraft.tests import data as test_data
from codraft.utils.qthelpers import QtTestEnv
from codraft.utils.tests import temporary_directory

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run image tools test scenario"""
    qttestenv = QtTestEnv()
    qttestenv.unattended = True
    with temporary_directory() as tmpdir:
        fname = osp.join(tmpdir, "test.json")
        with codraft_app_context() as win:
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
            print("Check metadata export <--> import features:")
            for key, value in orig_metadata.items():
                print(f"  Checking {key} key value...", end="")
                if isinstance(value, np.ndarray):
                    assert (value == ima.metadata[key]).all()
                else:
                    assert value == ima.metadata[key]
                print("OK")


if __name__ == "__main__":
    test()
