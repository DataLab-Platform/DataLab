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

# guitest: show

import os.path as osp

import numpy as np

from cdl.config import Conf
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests import data as test_data
from cdl.utils import tests


def get_metadata_param_number_after_reset():
    """Return metadata parameters number after reset"""
    def_ima_nb = len(Conf.view.get_def_dict("ima"))
    return def_ima_nb + 2  # +2 for metadata options (see BaseObj.get_metadata_option)


def test_metadata_io_unit():
    """Run image tools test scenario"""
    with execenv.context(unattended=True):
        with tests.CDLTemporaryDirectory() as tmpdir:
            fname = osp.join(tmpdir, "test.json")
            with cdltest_app_context() as win:
                panel = win.imagepanel
                ima = test_data.create_annotated_image()
                for mshape in test_data.create_resultshapes():
                    mshape.add_to(ima)
                panel.add_object(ima)
                orig_metadata = ima.metadata.copy()
                panel.export_metadata_from_file(fname)
                panel.delete_metadata()
                assert len(ima.metadata) == get_metadata_param_number_after_reset()
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
    test_metadata_io_unit()
