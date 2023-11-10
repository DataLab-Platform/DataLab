# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Denoise processing application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdlapp.param
from cdlapp.tests import cdl_app_context
from cdlapp.tests.data import get_test_image


def test():
    """Run denoise application test scenario"""
    with cdl_app_context() as win:
        win.showMaximized()
        panel = win.imagepanel
        panel.add_object(get_test_image("flower.npy"))
        proc = panel.processor
        proc.compute_all_denoise()
        panel.objview.select_groups([0])
        param = cdlapp.param.GridParam.create(cols=3)
        proc.distribute_on_grid(param)
        panel.add_label_with_title()


if __name__ == "__main__":
    test()
