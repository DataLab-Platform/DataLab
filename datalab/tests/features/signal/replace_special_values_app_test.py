# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Replace special values application test for signals.

This test verifies DataLab processor integration for ``replace_special_values``.
Algorithm correctness (including Inf handling) is covered by Sigima unit tests.
Only NaN values are injected here to avoid Qt plot instability with Inf data.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
import sigima.params
from sigima.enums import ReplacementStrategySignal as S
from sigima.objects import SignalObj, create_signal

from datalab.tests import datalab_test_app_context


def test_replace_special_values_signal_app():
    """Test signal replace special values through DataLab processor."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel

        # Use NaN-only data: Inf values destabilize Qt signal plot rendering
        x = np.arange(5, dtype=float)
        y = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        sig = create_signal("Signal with NaN values", x, y)
        panel.add_object(sig)
        panel.objview.select_objects([sig])

        param = sigima.params.ReplaceSpecialValuesSignalParam.create(
            nan_strategy=S.CONSTANT,
            nan_constant_value=10.0,
        )
        panel.processor.run_feature("replace_special_values", param, edit=False)

        result_objects = panel.objview.get_sel_objects()
        assert len(result_objects) == 1
        result = result_objects[0]
        assert isinstance(result, SignalObj)
        assert not np.isnan(result.y).any()
        np.testing.assert_array_equal(result.y, [1.0, 10.0, 3.0, 10.0, 5.0])


if __name__ == "__main__":
    test_replace_special_values_signal_app()
