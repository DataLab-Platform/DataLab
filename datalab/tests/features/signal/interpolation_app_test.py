# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Interpolation application test and X-array compatibility behavior.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
import sigima.params
from sigima.objects import SignalObj, create_signal

from datalab.tests import datalab_test_app_context


def test_interpolation_app():
    """Test interpolation feature in DataLab application.

    This test ensures that interpolation works correctly when signals have
    different X arrays (which is the whole point of interpolation).
    """
    with datalab_test_app_context() as win:
        panel = win.signalpanel

        # Create interpolation parameters
        param = sigima.params.InterpolationParam.create(method="linear")

        # Create X arrays
        x_dense = np.linspace(0, 10, 100)
        x_sparse = np.linspace(1, 9, 20)

        for x1, x2 in [(x_dense, x_sparse), (x_sparse, x_dense)]:
            context = f"Source X {'dense' if len(x1) > len(x2) else 'sparse'} → "
            context += f"Target X {'sparse' if len(x1) > len(x2) else 'dense'}"

            # Create source signal with dense sampling
            y1 = np.sin(x1)
            sig1 = create_signal("Source signal", x1, y1)
            panel.add_object(sig1)

            # Create target signal with sparse sampling and different range
            y2 = np.zeros_like(x2)  # Y values don't matter for interpolation target
            sig2 = create_signal("Target X values", x2, y2)
            panel.add_object(sig2)

            # This is the key test - interpolation should work even when X arrays differ
            # The X-array compatibility check should NOT interfere
            panel.objview.select_objects([sig1])
            panel.processor.run_feature("interpolate", sig2, param)

            # Verify the result
            result_objects = panel.objview.get_sel_objects()
            assert len(result_objects) == 1, (
                f"[{context}] Should have one interpolated result"
            )
            result = result_objects[0]
            assert isinstance(result, SignalObj), (
                f"[{context}] Result should be a SignalObj"
            )
            assert result.x.shape == sig2.x.shape, (
                f"[{context}] Result X should match target X shape"
            )
            assert np.allclose(result.x, sig2.x), (
                f"[{context}] Result X should match target X"
            )
            expected_y = np.interp(x2, x1, y1)
            assert np.allclose(result.y, expected_y, atol=1e-10), (
                f"[{context}] Result Y should be properly interpolated"
            )


def test_xarray_compatibility_still_works_for_other_operations():
    """Test that X-array compatibility still works for operations that need it.

    This test ensures that the fix for interpolation doesn't break X-array
    compatibility for other operations like addition.
    """
    with datalab_test_app_context() as win:
        panel = win.signalpanel

        # Create signals with different X arrays (should trigger compatibility check)
        x1 = np.linspace(0, 10, 100)
        y1 = np.ones_like(x1)
        sig1 = create_signal("Signal 1", x1, y1)
        panel.add_object(sig1)

        x2 = np.linspace(0, 10, 50)  # Different sampling
        y2 = np.ones_like(x2)
        sig2 = create_signal("Signal 2", x2, y2)
        panel.add_object(sig2)

        # Test addition (should still use X-array compatibility)
        panel.objview.select_objects([sig1, sig2])
        panel.processor.run_feature("addition")

        # Get the result
        result_objects = panel.objview.get_sel_objects()
        assert len(result_objects) == 1, "Should have one addition result"
        result = result_objects[0]

        # For addition, the X-array compatibility should have handled the different
        # arrays. The result should have consistent X values and proper Y values (2.0)
        assert np.allclose(result.y, 2.0, atol=1e-10), (
            "Addition result should be 2.0 (1.0 + 1.0)"
        )


if __name__ == "__main__":
    test_interpolation_app()
