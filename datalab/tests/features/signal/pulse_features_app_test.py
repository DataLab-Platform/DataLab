# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Pulse features application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from sigima.objects import create_signal_from_param
from sigima.tests.signal.pulse_unit_test import (
    create_test_square_params,
    create_test_step_params,
)

from datalab.tests import datalab_test_app_context


def test_pulse_features_app():
    """Pulse features application test."""
    with datalab_test_app_context() as win:
        panel = win.signalpanel
        s1 = create_signal_from_param(create_test_step_params())
        panel.add_object(s1)
        panel.processor.run_feature("extract_pulse_features")
        s2 = create_signal_from_param(create_test_square_params())
        panel.add_object(s2)
        panel.processor.run_feature("extract_pulse_features")


if __name__ == "__main__":
    test_pulse_features_app()
