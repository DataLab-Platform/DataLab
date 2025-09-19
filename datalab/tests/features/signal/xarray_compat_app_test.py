# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
X-array compatibility application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
from sigima.objects import GaussParam, SignalObj, create_signal_from_param

from datalab.config import Conf
from datalab.tests import datalab_test_app_context


def __check_addition_result(
    result: SignalObj, s_ref: SignalObj, context: str, coeff: float
) -> None:
    """Check that the Y data of the result is the same as the reference signal
    multiplied by coeff."""
    assert len(result.x) == len(s_ref.x), f"[{context}] Addition result length mismatch"
    assert np.allclose(result.y, coeff * s_ref.y, atol=1e-4), (
        f"[{context}] Addition result value mismatch"
    )


def __check_difference_result(
    result: SignalObj, s_ref: SignalObj, context: str
) -> None:
    """Check that the Y data of the result is zero."""
    assert len(result.x) == len(s_ref.x), (
        f"[{context}] Difference result length mismatch"
    )
    assert np.allclose(result.y, 0.0, atol=1e-4), (
        f"[{context}] Difference result value mismatch"
    )


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

        # Test with addition operation:
        with Conf.proc.xarray_compat_behavior.temp("interpolate"):
            # Select signals with the same number of points but different X ranges:
            panel.objview.select_objects([s_ref, s_same_nbp])
            panel.processor.run_feature("addition")
            __check_addition_result(
                panel.objview.get_sel_objects()[0], s_ref, "same_nbp", 2.0
            )

            # Select signals with different number of points but same X ranges:
            panel.objview.select_objects([s_ref, s_same_range])
            panel.processor.run_feature("addition")
            __check_addition_result(
                panel.objview.get_sel_objects()[0], s_ref, "same_range", 2.0
            )

            # Select signals with different number of points and different X ranges:
            panel.objview.select_objects([s_ref, s_different])
            panel.processor.run_feature("addition")
            __check_addition_result(
                panel.objview.get_sel_objects()[0], s_ref, "different", 2.0
            )

            # Select all signals and add them:
            panel.objview.select_objects([s_ref, s_same_nbp, s_same_range, s_different])
            panel.processor.run_feature("addition")
            __check_addition_result(
                panel.objview.get_sel_objects()[0], s_ref, "all", 4.0
            )

            # Select all signals and average them:
            panel.objview.select_objects([s_ref, s_same_nbp, s_same_range, s_different])
            panel.processor.run_feature("average")
            __check_addition_result(
                panel.objview.get_sel_objects()[0], s_ref, "all", 1.0
            )

        # Test with subtraction operation:
        with Conf.proc.xarray_compat_behavior.temp("interpolate"):
            # Select signals with the same number of points but different X ranges:
            panel.objview.select_objects([s_ref])
            panel.processor.run_feature("difference", obj2=s_same_nbp)
            __check_difference_result(
                panel.objview.get_sel_objects()[0], s_ref, "same_nbp"
            )

            # Select signals with different number of points but same X ranges:
            panel.objview.select_objects([s_ref])
            panel.processor.run_feature("difference", obj2=s_same_range)
            __check_difference_result(
                panel.objview.get_sel_objects()[0], s_ref, "same_range"
            )

            # Select signals with different number of points and different X ranges:
            panel.objview.select_objects([s_ref])
            panel.processor.run_feature("difference", obj2=s_different)
            __check_difference_result(
                panel.objview.get_sel_objects()[0], s_ref, "different"
            )


if __name__ == "__main__":
    test_xarray_compatibility_app()
