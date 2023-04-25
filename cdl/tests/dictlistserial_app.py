# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Testing (de)serialization of Dictionnary/List inside object metadata
"""

import os.path as osp

import numpy as np
from skimage.data import human_mitosis  # pylint: disable=no-name-in-module

from cdl.core.model.image import create_image
from cdl.env import execenv
from cdl.tests import cdl_app_context
from cdl.utils.tests import temporary_directory

SHOW = False  # Show test in GUI-based test launcher


def __compare_metadata(dict1, dict2):
    """Compare metadata dictionaries without private elements"""
    dict_a, dict_b = dict1.copy(), dict2.copy()
    for dict_ in (dict_a, dict_b):
        for key in list(dict_.keys()):
            if key.startswith("__"):
                dict_.pop(key)
    return str(dict_a) == str(dict_b)


def test():
    """Dictionnary/List in metadata (de)serialization test"""
    execenv.unattended = True
    with temporary_directory() as tmpdir:
        with cdl_app_context(console=False) as win:
            panel = win.imagepanel

            data = human_mitosis()
            image = create_image("Test image with peaks", data)
            image.metadata["tata"] = {
                "lkl": 2,
                "tototo": 3,
                "zzzz": "lklk",
                "d": {"lkl": 2, "tototo": 3, "zzzz": "lklk"},
            }
            image.metadata["toto"] = [
                np.array([[1, 2], [-3, 0]]),
                np.array([[1, 2], [-3, 0], [99, 241]]),
            ]
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
    test()
