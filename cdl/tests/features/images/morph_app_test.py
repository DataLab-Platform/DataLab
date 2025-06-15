# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Morphology processing application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima_.param
from cdl.tests import cdltest_app_context
from cdl.tests.data import get_test_image


def test_morphology():
    """Run morphology application test scenario"""
    with cdltest_app_context() as win:
        win.showMaximized()
        panel = win.imagepanel
        panel.add_object(get_test_image("flower.npy"))
        proc = panel.processor
        param = sigima_.param.MorphologyParam.create(radius=10)
        proc.compute_all_morphology(param)
        panel.objview.select_groups()
        param = sigima_.param.GridParam.create(cols=4)
        proc.distribute_on_grid(param)
        panel.add_label_with_title()


if __name__ == "__main__":
    test_morphology()
