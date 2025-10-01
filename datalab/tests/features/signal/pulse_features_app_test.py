# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Pulse features application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from sigima.objects import SignalObj, TableKind, TableResult, create_signal_from_param
from sigima.tests.helpers import check_scalar_result
from sigima.tests.signal.pulse_unit_test import (
    create_test_square_params,
    create_test_step_params,
)

from datalab.adapters_metadata import TableAdapter
from datalab.tests import datalab_test_app_context


def __check_table(obj: SignalObj) -> TableResult:
    """Check that the object has a pulse features table."""
    tables = list(TableAdapter.iterate_from_obj(obj))
    assert len(tables) == 1
    table = tables[0].result
    assert table.kind == TableKind.PULSE_FEATURES
    return table


def test_pulse_features_app():
    """Pulse features application test."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel

        # Add first signal and extract features
        s1 = create_signal_from_param(create_test_step_params())
        panel.add_object(s1)
        panel.processor.run_feature("extract_pulse_features")

        # Check that features are extracted
        table1 = __check_table(s1)
        assert table1["signal_shape"][0] == "step"
        assert table1["polarity"][0] == 1.0
        for name, expected_value in (("amplitude", 4.9), ("rise_time", 1.5)):
            check_scalar_result(name, table1[name][0], expected_value, rtol=1e-2)

        # Add second signal and extract features
        s2 = create_signal_from_param(create_test_square_params())
        panel.add_object(s2)
        panel.processor.run_feature("extract_pulse_features")

        # Check that features are extracted
        table2 = __check_table(s2)
        assert table2["signal_shape"][0] == "square"
        assert table2["polarity"][0] == 1.0
        for name, expected_value in (
            ("amplitude", 4.94),
            ("rise_time", 1.60),
            ("fall_time", 3.95),
            ("fwhm", 5.49),
        ):
            check_scalar_result(name, table2[name][0], expected_value, rtol=1e-2)


if __name__ == "__main__":
    test_pulse_features_app()
