# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Pulse features application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from sigima.objects import (
    SignalObj,
    TableKind,
    TableResult,
    create_signal_from_param,
    create_signal_roi,
)
from sigima.tests.helpers import check_scalar_result
from sigima.tests.signal.pulse_unit_test import (
    create_test_square_params,
    create_test_step_params,
)

from datalab.adapters_metadata import TableAdapter
from datalab.env import execenv
from datalab.gui.panel.signal import SignalPanel
from datalab.tests import datalab_test_app_context


def __check_table(obj: SignalObj) -> TableResult:
    """Check that the object has a pulse features table."""
    tables = list(TableAdapter.iterate_from_obj(obj))
    assert len(tables) == 1
    table = tables[0].result
    assert table.kind == TableKind.PULSE_FEATURES
    return table


def __add_signal_and_check_pulse_features(
    panel: SignalPanel, obj: SignalObj, assertions: dict[str, float]
) -> None:
    """Add signal to the application and check that pulse features are extracted."""
    panel.add_object(obj)
    panel.processor.run_feature("extract_pulse_features")
    table = __check_table(obj)
    for name, expected_value in assertions.items():
        if isinstance(expected_value, str):
            assert table[name][0] == expected_value
        else:
            check_scalar_result(name, table[name][0], expected_value, rtol=1e-2)


def test_pulse_features_app():
    """Pulse features application test."""
    execenv.unattended = True
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel

        # Add first signal and extract features
        s1 = create_signal_from_param(create_test_step_params())
        __add_signal_and_check_pulse_features(
            panel,
            s1,
            {
                "signal_shape": "step",
                "polarity": 1.0,
                "amplitude": 4.9,
                "rise_time": 1.5,
            },
        )

        # Add second signal and extract features
        s2 = create_signal_from_param(create_test_square_params())
        __add_signal_and_check_pulse_features(
            panel,
            s2,
            {
                "signal_shape": "square",
                "polarity": 1.0,
                "amplitude": 4.94,
                "rise_time": 1.60,
                "fall_time": 3.95,
                "fwhm": 5.49,
            },
        )

        # Select the two signals and show the results table
        panel.objview.select_objects([s1, s2])
        panel.show_results()

        # Define a ROI, just to test that it works (⚠️ it makes no sense here):
        # it tests that the "comparison rows" features work correctly with ROIs.
        # TODO: Maybe we should test this part in a more appropriate place, like in
        # a test for the "statistics" feature.
        roi = create_signal_roi([[0.650227, 5.8], [1.1596, 9.09509]])
        s1.roi = roi
        s2.roi = roi.copy()
        panel.processor.run_feature("extract_pulse_features")


if __name__ == "__main__":
    test_pulse_features_app()
