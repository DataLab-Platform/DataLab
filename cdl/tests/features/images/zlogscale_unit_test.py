# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Z-log scale test

Testing z-log scale tool feature.
"""

# guitest: show

import numpy as np
from guidata.qthelpers import qt_app_context
from plotpy.builder import make

from cdl import patch  # pylint: disable=unused-import  # noqa: F401
from cdl.tests.data import create_2d_steps_data
from cdl.utils.vistools import view_image_items


def test_zlogscale():
    """Z-log scale test"""
    with qt_app_context():
        data = create_2d_steps_data(1024, width=256, dtype=np.int32)
        item = make.image(data)
        view_image_items([item], title="Z-log scale test")
        item.set_zaxis_log_state(True)  # pylint: disable=no-member


if __name__ == "__main__":
    test_zlogscale()
