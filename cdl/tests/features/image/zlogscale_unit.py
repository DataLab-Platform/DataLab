# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Z-log scale test

Testing z-log scale tool feature.
"""

# guitest: show

import numpy as np
from guiqwt.builder import make

from cdl import patch  # pylint: disable=unused-import
from cdl.tests.data import create_2d_steps_data
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.vistools import view_image_items


def zlogscale_test():
    """Z-log scale test"""
    with qt_app_context():
        data = create_2d_steps_data(1024, width=256, dtype=np.int32)
        item = make.image(data)
        view_image_items([item], title="Z-log scale test")
        item.set_zaxis_log_state(True)  # pylint: disable=no-member


if __name__ == "__main__":
    zlogscale_test()
