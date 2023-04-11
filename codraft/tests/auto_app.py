# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT mainwindow automation test:

  - Saving current project (h5 file)
  - Removing all objects
  - Opening another project (h5 file)
  - Access current image metadata
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import os.path as osp

from codraft.core.gui.processor.base import ClipParam
from codraft.env import execenv
from codraft.tests import codraft_app_context
from codraft.tests.data import create_test_image2, create_test_signal1
from codraft.utils.tests import temporary_directory

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Run ROI unit test scenario"""
    with temporary_directory() as tmpdir:
        size = 500
        with codraft_app_context(console=False) as win:
            # === Creating two test signals
            panel = win.signalpanel
            sig1 = create_test_signal1(size)
            panel.add_object(sig1)
            panel.processor.compute_derivative()
            # === Creating two test images
            panel = win.imagepanel
            ima1 = create_test_image2(size, with_annotations=True)
            panel.add_object(ima1)
            param = ClipParam()
            param.value = ima1.data.mean()
            panel.processor.compute_clip(param)
            # === Saving project
            fname = osp.join(tmpdir, "test.h5")
            win.save_to_h5_file(fname)
            # === Removing all objects
            for panel in win.panels:
                panel.remove_all_objects()
            # === Reopening previously saved project
            win.open_h5_files([fname], import_all=True, reset_all=True)
            # === Accessing object metadata
            obj = win.imagepanel.objlist[1]
            execenv.print(f"Image '{obj.title}':")
            for key, value in obj.metadata.items():
                execenv.print(f'  metadata["{key}"] = {value}')


if __name__ == "__main__":
    test()
