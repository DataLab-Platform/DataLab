# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Replace special values application test for images.

This test verifies DataLab processor integration for ``replace_special_values``.
Algorithm correctness (including Inf handling) is covered by Sigima unit tests.
Only NaN values are injected here to avoid Qt plot instability with Inf data.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import numpy as np
import sigima.params
from sigima.enums import ReplacementStrategyImage as S
from sigima.objects import ImageObj, create_image

from datalab.tests import datalab_test_app_context


def test_replace_special_values_image_app():
    """Test image replace special values through DataLab processor."""
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel

        # Use NaN-only data: Inf values destabilize Qt image plot rendering/teardown
        data = np.array(
            [
                [1.0, np.nan, 3.0],
                [4.0, 5.0, np.nan],
                [7.0, 8.0, 9.0],
            ]
        )
        image = create_image("Image with NaN values", data)
        panel.add_object(image)
        panel.objview.select_objects([image])

        param = sigima.params.ReplaceSpecialValuesImageParam.create(
            nan_strategy=S.CONSTANT,
            nan_constant_value=10.0,
        )
        panel.processor.run_feature("replace_special_values", param, edit=False)

        result_objects = panel.objview.get_sel_objects()
        assert len(result_objects) == 1
        result = result_objects[0]
        assert isinstance(result, ImageObj)
        assert not np.isnan(result.data).any()
        np.testing.assert_array_equal(
            result.data,
            np.array(
                [
                    [1.0, 10.0, 3.0],
                    [4.0, 5.0, 10.0],
                    [7.0, 8.0, 9.0],
                ]
            ),
        )


if __name__ == "__main__":
    test_replace_special_values_image_app()
