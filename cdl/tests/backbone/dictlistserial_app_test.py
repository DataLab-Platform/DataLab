# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Testing (de)serialization of Dictionnary/List inside object metadata
"""

# guitest: show

import os.path as osp

import numpy as np
from sigima.obj import create_image
from sigima.tests.data import get_test_image
from sigima.tests.helpers import WorkdirRestoringTempDir, compare_metadata

from cdl.env import execenv
from cdl.tests import cdltest_app_context


def test_dict_serialization():
    """Dictionnary/List in metadata (de)serialization test"""
    with execenv.context(unattended=True):
        with WorkdirRestoringTempDir() as tmpdir:
            with cdltest_app_context(console=False) as win:
                panel = win.imagepanel

                data = get_test_image("flower.npy").data
                image = create_image("Test image with peaks", data)
                image.metadata["tata"] = {
                    "lkl": 2,
                    "tototo": 3,
                    "arrdata": np.array([0, 1, 2, 3, 4, 5]),
                    "zzzz": "lklk",
                    "bool": True,
                    "float": 1.234,
                    "list": [1, 2.5, 3, "str", False, 5],
                    "d": {
                        "lkl": 2,
                        "tototo": 3,
                        "zzzz": "lklk",
                        "bool": True,
                        "float": 1.234,
                        "list": [
                            1,
                            2.5,
                            3,
                            "str",
                            False,
                            5,
                            {"lkl": 2, "l": [1, 2, 3]},
                        ],
                    },
                }
                image.metadata["toto"] = [
                    np.array([[1, 2], [-3, 0]]),
                    np.array([[1, 2], [-3, 0], [99, 241]]),
                ]
                image.metadata["array"] = np.array([-5, -4, -3, -2, -1])
                panel.add_object(image)
                fname = osp.join(tmpdir, "test.h5")
                win.save_to_h5_file(fname)
                win.reset_all()
                win.open_h5_files([fname], import_all=True)
                execenv.print("Dictionary/List (de)serialization:")
                oids = panel.objmodel.get_object_ids()
                first_image = panel.objmodel[oids[0]]
                assert compare_metadata(image.metadata, first_image.metadata.copy())


if __name__ == "__main__":
    test_dict_serialization()
