# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Edges processing application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdl.param
from cdl.tests import cdltest_app_context
from cdl.tests.data import get_test_image


def test_edges():
    """Run edges processing application test scenario"""
    with cdltest_app_context() as win:
        win.showMaximized()
        panel = win.imagepanel
        panel.add_object(get_test_image("flower.npy"))
        proc = panel.processor
        proc.compute_all_edges()
        panel.objview.select_groups()
        param = cdl.param.GridParam.create(cols=4)
        proc.distribute_on_grid(param)
        panel.add_label_with_title()


if __name__ == "__main__":
    test_edges()
