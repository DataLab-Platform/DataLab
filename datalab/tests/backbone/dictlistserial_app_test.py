# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Testing (de)serialization of Dictionnary/List inside object metadata
"""

# guitest: show

import os.path as osp

from sigima.tests.data import create_test_image_with_metadata

from datalab.env import execenv
from datalab.tests import datalab_test_app_context, helpers


def test_dict_serialization():
    """Dictionnary/List in metadata (de)serialization test"""
    with execenv.context(unattended=True):
        with helpers.WorkdirRestoringTempDir() as tmpdir:
            with datalab_test_app_context(console=False) as win:
                panel = win.imagepanel
                image = create_test_image_with_metadata()
                panel.add_object(image)
                fname = osp.join(tmpdir, "test.h5")
                win.save_to_h5_file(fname)
                win.reset_all()
                win.open_h5_files([fname], import_all=True)
                execenv.print("Dictionary/List (de)serialization:")
                oids = panel.objmodel.get_object_ids()
                first_image = panel.objmodel[oids[0]]
                assert helpers.compare_metadata(
                    image.metadata, first_image.metadata.copy()
                )


if __name__ == "__main__":
    test_dict_serialization()
