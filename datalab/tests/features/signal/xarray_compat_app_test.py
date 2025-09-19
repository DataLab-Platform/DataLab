# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
X-array compatibility application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from sigima.objects import GaussParam, create_signal_from_param

from datalab.config import Conf
from datalab.tests import datalab_test_app_context


def test_xarray_compatibility_app():
    """X-array compatibility application test.

    This test aims at verifying that the X-array compatibility feature works as expected
    in the application context.
    """
    with datalab_test_app_context() as win:
        panel = win.signalpanel

        # Reference parameters:
        p_ref = GaussParam.create(size=500, xmin=-10, xmax=10)

        # Parameters with the same number of points but different X ranges:
        p_same_nbp = GaussParam.create(size=500, xmin=-5, xmax=5)

        # Parameters with different number of points but same X ranges:
        p_same_range = GaussParam.create(size=1000, xmin=-10, xmax=10)

        # Parameters with different number of points and different X ranges:
        p_different = GaussParam.create(size=1000, xmin=-5, xmax=5)

        panel.add_object(s_ref := create_signal_from_param(p_ref))
        panel.add_object(s_same_nbp := create_signal_from_param(p_same_nbp))
        panel.add_object(s_same_range := create_signal_from_param(p_same_range))
        panel.add_object(s_different := create_signal_from_param(p_different))

        # Select signals with the same number of points but different X ranges:
        panel.objview.select_objects([s_ref, s_same_nbp])
        # Try to compute the sum:
        panel.processor.run_feature("addition")

        # Select signals with different number of points but same X ranges:
        panel.objview.select_objects([s_ref, s_same_range])
        # Try to compute the sum:
        panel.processor.run_feature("addition")

        # Select signals with different number of points and different X ranges:
        panel.objview.select_objects([s_ref, s_different])
        # Try to compute the sum:
        panel.processor.run_feature("addition")


if __name__ == "__main__":
    test_xarray_compatibility_app()
