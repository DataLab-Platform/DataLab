# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Denoise processing application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.param
from sigima.tests.data import get_test_image

from datalab.tests import cdltest_app_context


def test_denoise():
    """Run denoise application test scenario"""
    with cdltest_app_context() as win:
        win.showMaximized()
        panel = win.imagepanel
        panel.add_object(get_test_image("flower.npy"))
        proc = panel.processor
        proc.compute_all_denoise()
        panel.objview.select_groups()
        param = sigima.param.GridParam.create(cols=3)
        proc.distribute_on_grid(param)
        panel.add_label_with_title()


if __name__ == "__main__":
    test_denoise()
