# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal offset correction unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import numpy as np
import pytest
from guidata.qthelpers import exec_dialog, qt_app_context

import sigima.signal as ss
from cdl.obj import ROI1DParam
from cdl.tests.data import create_paracetamol_signal
from cdl.utils.vistools import view_curve_items
from cdl.widgets.signalbaseline import SignalBaselineDialog


def test_signal_offset_correction_interactive() -> None:
    """Signal offset correction interactive test."""
    with qt_app_context():
        s1 = create_paracetamol_signal()
        dlg = SignalBaselineDialog(s1)
        if exec_dialog(dlg):
            param = ROI1DParam()
            param.xmin, param.xmax = dlg.get_x_range()
            s2 = ss.offset_correction(s1, param)
            view_curve_items(
                [s1.make_item(), s2.make_item()], title="Signal offset correction"
            )


@pytest.mark.validation
def test_signal_offset_correction() -> None:
    """Signal offset correction validation test."""
    s1 = create_paracetamol_signal()
    param = ROI1DParam.create(xmin=10.0, xmax=12.0)
    s2 = ss.offset_correction(s1, param)

    # Check that the offset correction has been applied
    imin, imax = np.searchsorted(s1.x, [param.xmin, param.xmax])
    offset = np.mean(s1.y[imin:imax])
    assert np.allclose(s2.y, s1.y - offset), "Offset correction failed"


if __name__ == "__main__":
    test_signal_offset_correction_interactive()
    test_signal_offset_correction()
