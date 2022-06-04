# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Z-log scale test

Testing z-log scale tool feature.
"""

import numpy as np
from guiqwt.builder import make

from codraft import patch  # pylint: disable=unused-import
from codraft.tests.data import create_2d_steps_data
from codraft.utils.qthelpers import qt_app_context
from codraft.utils.vistools import view_image_items

SHOW = True  # Show test in GUI-based test launcher


def zlogscale_test():
    """Z-log scale test"""
    with qt_app_context():
        data = create_2d_steps_data(1024, width=256, dtype=np.int32)
        item = make.image(data)
        view_image_items([item], title="Z-log scale test")
        item.set_zaxis_log_state(True)  # pylint: disable=no-member


if __name__ == "__main__":
    zlogscale_test()
