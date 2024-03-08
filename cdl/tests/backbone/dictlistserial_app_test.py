# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Testing (de)serialization of Dictionnary/List inside object metadata
"""

# guitest: show

import os.path as osp

import numpy as np

from cdl.env import execenv
from cdl.obj import create_image
from cdl.tests import cdltest_app_context
from cdl.tests.data import get_test_image
from cdl.utils.tests import CDLTemporaryDirectory


def __compare_metadata(dict1, dict2):
    """Compare metadata dictionaries without private elements"""
    dict_a, dict_b = dict1.copy(), dict2.copy()
    for dict_ in (dict_a, dict_b):
        for key in list(dict_.keys()):
            if key.startswith("__"):
                dict_.pop(key)
    same = True
    for key in dict_a:
        if key not in dict_b:
            same = False
            break
        if isinstance(dict_a[key], dict):
            same = same and __compare_metadata(dict_a[key], dict_b[key])
        else:
            same_value = str(dict_a[key]) == str(dict_b[key])
            if not same_value:
                print(f"Different values for key {key}: {dict_a[key]} != {dict_b[key]}")
            same = same and same_value
    return same


def test_dict_serialization():
    """Dictionnary/List in metadata (de)serialization test"""
    with execenv.context(unattended=True):
        with CDLTemporaryDirectory() as tmpdir:
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
                execenv.print("Dictionary/List (de)serialization: ", end="")
                oids = panel.objmodel.get_object_ids()
                first_image = panel.objmodel[oids[0]]
                assert __compare_metadata(image.metadata, first_image.metadata.copy())
                execenv.print("OK")


if __name__ == "__main__":
    test_dict_serialization()
