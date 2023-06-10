# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Morphology processing application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdl.param
from cdl.tests import cdl_app_context
from cdl.tests.data import get_test_image


def test():
    """Run morphology application test scenario"""
    with cdl_app_context() as win:
        win.showMaximized()
        panel = win.imagepanel
        panel.add_object(get_test_image("flower.npy"))
        proc = panel.processor
        param = cdl.param.MorphologyParam()
        param.radius = 10
        proc.compute_all_morphology(param)
        panel.objview.select_groups([0])
        param = cdl.param.GridParam()
        param.cols = 4
        proc.distribute_on_grid(param)
        panel.add_label_with_title()


if __name__ == "__main__":
    test()
