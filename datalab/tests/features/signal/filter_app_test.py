# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Frequential filtering application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima.params
from sigima.objects import create_signal_from_param
from sigima.tests.helpers import check_array_result
from sigima.tests.signal.pulse.pulse_unit_test import create_test_square_params

from datalab.tests import datalab_test_app_context


def test_signal_freq_filter_app():
    """Signal frequency filtering application test."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        s1 = create_signal_from_param(create_test_square_params())
        panel.add_object(s1)
        for feature, paramclass in (
            ("lowpass", sigima.params.LowPassFilterParam),
            ("highpass", sigima.params.HighPassFilterParam),
            ("bandstop", sigima.params.BandStopFilterParam),
        ):
            for zero_padding in (True, False):
                panel.objview.select_objects([1])  # Select the first signal
                param = paramclass.create(method="brickwall", zero_padding=zero_padding)
                param.update_from_obj(s1)
                panel.processor.run_feature(feature, param)

                s2 = panel.objview.get_sel_objects()[0]
                check_array_result(
                    f"{feature} filter output X data (zero_padding={zero_padding})",
                    s2.x,
                    s1.x,
                )


if __name__ == "__main__":
    test_signal_freq_filter_app()
