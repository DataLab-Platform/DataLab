# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal offset correction unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import numpy as np
import pytest

import sigima_.computation.signal as sigima_signal
import sigima_.obj
from sigima_.tests.data import create_paracetamol_signal


@pytest.mark.validation
def test_signal_offset_correction() -> None:
    """Signal offset correction validation test."""
    s1 = create_paracetamol_signal()
    param = sigima_.obj.ROI1DParam.create(xmin=10.0, xmax=12.0)
    s2 = sigima_signal.offset_correction(s1, param)

    # Check that the offset correction has been applied
    imin, imax = np.searchsorted(s1.x, [param.xmin, param.xmax])
    offset = np.mean(s1.y[imin:imax])
    assert np.allclose(s2.y, s1.y - offset), "Offset correction failed"


if __name__ == "__main__":
    test_signal_offset_correction()
