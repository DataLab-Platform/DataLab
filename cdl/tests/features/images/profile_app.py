# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Profile extraction test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdl.param
from cdl.tests import cdltest_app_context
from cdl.tests.data import get_test_image


def test_profile():
    """Run profile extraction test scenario"""
    with cdltest_app_context() as win:
        panel = win.imagepanel
        panel.add_object(get_test_image("flower.npy"))
        proc = panel.processor
        for direction, row, col in (
            ("horizontal", 102, 131),
            ("vertical", 102, 131),
        ):
            profparam = cdl.param.ProfileParam.create(
                direction=direction, row=row, col=col
            )
            proc.compute_profile(profparam)
        for direction, row1, col1, row2, col2 in (
            ("horizontal", 10, 10, 102, 131),
            ("vertical", 10, 10, 102, 131),
        ):
            avgprofparam = cdl.param.AverageProfileParam.create(
                direction=direction,
                row1=row1,
                col1=col1,
                row2=row2,
                col2=col2,
            )
            proc.compute_average_profile(avgprofparam)


if __name__ == "__main__":
    test_profile()
