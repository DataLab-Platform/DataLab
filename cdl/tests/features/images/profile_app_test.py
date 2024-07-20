# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction test
=======================

Testing the profile extraction features of the image panel:

- Compute a profile along a horizontal line
- Compute a profile along a vertical line
- Compute an average profile between two points along a horizontal line
- Compute an average profile between two points along a vertical line
- Compute a radial profile
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import cdl.param
from cdl.tests import cdltest_app_context
from cdl.tests.data import create_noisygauss_image, get_test_image


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
            profparam = cdl.param.LineProfileParam.create(
                direction=direction, row=row, col=col
            )
            proc.compute_line_profile(profparam)
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
        segprofparam = cdl.param.SegmentProfileParam.create(
            row1=10, col1=10, row2=102, col2=131
        )
        proc.compute_segment_profile(segprofparam)
        image2 = create_noisygauss_image(center=(0.0, 0.0), add_annotations=False)
        panel.add_object(image2)
        for center, x0, y0 in (
            (None, None, None),
            ("centroid", None, None),
            ("center", None, None),
            ("manual", 800, 900),
        ):
            if center is None:
                proc.compute_radial_profile()
            else:
                param = cdl.param.RadialProfileParam.create(center=center, x0=x0, y0=y0)
                proc.compute_radial_profile(param)


if __name__ == "__main__":
    test_profile()
