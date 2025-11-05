# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab-specific title formatting configuration tests

This module verifies that DataLab correctly configures the Sigima title formatting
system. The actual functionality of the title formatting system is comprehensively
tested in Sigima's test suite (sigima.tests.common.title_formatting_unit_test).

This test suite only verifies:
  - DataLab uses PlaceholderTitleFormatter by default (required for DataLab)
  - DataLab-specific complex parameter patterns work correctly

For comprehensive tests of the title formatting system itself, see:
  sigima/tests/common/title_formatting_unit_test.py
"""

from __future__ import annotations

import pytest
from sigima import create_signal
from sigima.proc.base import dst_1_to_1
from sigima.proc.title_formatting import (
    PlaceholderTitleFormatter,
    get_default_title_formatter,
    set_default_title_formatter,
)


class TestDataLabTitleFormattingConfiguration:
    """Test suite verifying DataLab's title formatting configuration.

    Note: Comprehensive tests of the title formatting system itself are in
    sigima/tests/common/title_formatting_unit_test.py. These tests only verify
    DataLab-specific configuration and usage patterns.
    """

    def test_datalab_uses_placeholder_formatter_by_default(self):
        """Verify DataLab is configured with PlaceholderTitleFormatter by default.

        This is critical for DataLab's workflow, as placeholder titles like
        "wiener({0})" are later resolved by DataLab's patch_title_with_ids()
        function to create titles like "wiener(s001)".
        """
        current_formatter = get_default_title_formatter()
        assert isinstance(current_formatter, PlaceholderTitleFormatter), (
            "DataLab must use PlaceholderTitleFormatter for proper title resolution"
        )

    def test_complex_datalab_parameter_patterns(self):
        """Test complex parameter strings typical in DataLab.

        DataLab often uses complex, multi-valued parameter strings in operation
        titles. This test verifies these patterns work correctly.
        """
        original_formatter = get_default_title_formatter()
        try:
            set_default_title_formatter(PlaceholderTitleFormatter())

            signal = create_signal("Test", x=[1, 2, 3], y=[4, 5, 6])

            # Test DataLab's typical complex parameter patterns
            complex_params = [
                "sigma=1.5,mode=constant",
                "window=hamming,nperseg=256,noverlap=128",
                "method=butterworth,order=4,critical_freq=0.1",
                "roi=[10,20,30,40],background=auto",
            ]

            for params in complex_params:
                result = dst_1_to_1(signal, "filter", params)
                expected = f"filter({{0}})|{params}"
                assert result.title == expected, (
                    f"Complex parameter '{params}' not preserved in title"
                )

        finally:
            set_default_title_formatter(original_formatter)


if __name__ == "__main__":
    pytest.main([__file__])
