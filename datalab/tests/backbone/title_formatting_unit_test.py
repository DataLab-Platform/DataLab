# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Title formatting system unit tests for DataLab

Testing that DataLab correctly uses the Sigima title formatting system to generate
placeholder-based titles that are compatible with DataLab's object naming and
processing workflow.

This test verifies:
  - DataLab uses PlaceholderTitleFormatter by default
  - Placeholder titles are generated correctly for various operations
  - Title resolution works with DataLab's object naming conventions
  - Integration with DataLab's signal and image processing functions
"""

from __future__ import annotations

import pytest
from sigima import create_image, create_signal
from sigima.proc.base import dst_1_to_1, dst_2_to_1, dst_n_to_1
from sigima.proc.title_formatting import (
    PlaceholderTitleFormatter,
    SimpleTitleFormatter,
    get_default_title_formatter,
    set_default_title_formatter,
)


class TestDataLabTitleFormatting:
    """Test suite for DataLab title formatting configuration."""

    def test_datalab_uses_placeholder_formatter_by_default(self):
        """Test that DataLab uses PlaceholderTitleFormatter by default."""
        # DataLab should be configured with PlaceholderTitleFormatter during startup
        current_formatter = get_default_title_formatter()
        assert isinstance(current_formatter, PlaceholderTitleFormatter)

    def test_placeholder_title_generation_for_signals(self):
        """Test placeholder title generation for signal processing operations."""
        # Ensure we're using the placeholder formatter
        original_formatter = get_default_title_formatter()
        try:
            set_default_title_formatter(PlaceholderTitleFormatter())

            # Test 1-to-1 operations (typical signal processing)
            signal = create_signal("Test Signal", x=[1, 2, 3, 4], y=[1, 4, 9, 16])

            # Test basic operation without suffix
            result = dst_1_to_1(signal, "gaussian_filter")
            assert result.title == "gaussian_filter({0})"

            # Test operation with parameters
            result = dst_1_to_1(signal, "wiener", "noise_var=0.1")
            assert result.title == "wiener({0})|noise_var=0.1"

            # Test mathematical operations
            result = dst_1_to_1(signal, "abs")
            assert result.title == "abs({0})"

        finally:
            set_default_title_formatter(original_formatter)

    def test_placeholder_title_generation_for_images(self):
        """Test placeholder title generation for image processing operations."""
        original_formatter = get_default_title_formatter()
        try:
            set_default_title_formatter(PlaceholderTitleFormatter())

            # Create test image
            import numpy as np

            data = np.random.rand(10, 10)
            image = create_image("Test Image", data=data)

            # Test image processing operation
            result = dst_1_to_1(image, "gaussian_filter", "sigma=2.0")
            assert result.title == "gaussian_filter({0})|sigma=2.0"

            # Test edge detection
            result = dst_1_to_1(image, "sobel")
            assert result.title == "sobel({0})"

        finally:
            set_default_title_formatter(original_formatter)

    def test_n_to_1_operations(self):
        """Test placeholder title generation for n-to-1 operations."""
        original_formatter = get_default_title_formatter()
        try:
            set_default_title_formatter(PlaceholderTitleFormatter())

            # Create multiple test signals
            signals = [
                create_signal(f"Signal{i}", x=[1, 2, 3], y=[i, i + 1, i + 2])
                for i in range(1, 4)
            ]

            # Test sum operation (typical for DataLab)
            result = dst_n_to_1(signals, "sum")
            assert result.title == "sum({0}, {1}, {2})"

            # Test with parameters
            result = dst_n_to_1(signals, "average", "weighted=True")
            assert result.title == "average({0}, {1}, {2})|weighted=True"

        finally:
            set_default_title_formatter(original_formatter)

    def test_2_to_1_operations(self):
        """Test placeholder title generation for 2-to-1 operations."""
        original_formatter = get_default_title_formatter()
        try:
            set_default_title_formatter(PlaceholderTitleFormatter())

            # Create test signals
            sig1 = create_signal("Signal1", x=[1, 2, 3], y=[1, 2, 3])
            sig2 = create_signal("Signal2", x=[1, 2, 3], y=[4, 5, 6])

            # Test arithmetic operations (common in DataLab)
            result = dst_2_to_1(sig1, sig2, "subtract")
            assert result.title == "subtract({0}, {1})"

            result = dst_2_to_1(sig1, sig2, "divide", "method=safe")
            assert result.title == "divide({0}, {1})|method=safe"

        finally:
            set_default_title_formatter(original_formatter)

    def test_title_resolution_with_mock_datalab_objects(self):
        """Test title resolution using DataLab-style object naming."""
        original_formatter = get_default_title_formatter()
        try:
            formatter = PlaceholderTitleFormatter()
            set_default_title_formatter(formatter)

            # Create signals that simulate DataLab object structure
            # (DataLab objects would have short_id attributes, but we test fallback)
            sig1 = create_signal("s001", x=[1, 2], y=[3, 4])
            sig2 = create_signal("s002", x=[1, 2], y=[5, 6])

            # Test placeholder resolution (uses obj1, obj2 fallbacks)
            result = formatter.resolve_placeholder_title("wiener({0})", [sig1])
            assert result == "wiener(obj1)"

            result = formatter.resolve_placeholder_title("add({0}, {1})", [sig1, sig2])
            assert result == "add(obj1, obj2)"

            # Test with parameters preserved
            result = formatter.resolve_placeholder_title(
                "gaussian_filter({0})|sigma=1.5", [sig1]
            )
            assert result == "gaussian_filter(obj1)|sigma=1.5"

        finally:
            set_default_title_formatter(original_formatter)


class TestDataLabTitleFormattingIntegration:
    """Test suite for DataLab title formatting integration scenarios."""

    def test_typical_datalab_workflow(self):
        """Test title formatting in a typical DataLab processing workflow."""
        original_formatter = get_default_title_formatter()
        try:
            # Ensure DataLab configuration
            set_default_title_formatter(PlaceholderTitleFormatter())

            # Simulate typical DataLab signal processing workflow
            # 1. Create initial signal
            original_signal = create_signal(
                "Original", x=[1, 2, 3, 4, 5], y=[1, 4, 9, 16, 25]
            )

            # 2. Apply smoothing filter
            smoothed = dst_1_to_1(original_signal, "gaussian_filter", "sigma=1.0")
            assert smoothed.title == "gaussian_filter({0})|sigma=1.0"

            # 3. Apply noise reduction
            denoised = dst_1_to_1(smoothed, "wiener", "noise_var=0.1")
            assert denoised.title == "wiener({0})|noise_var=0.1"

            # 4. Combine with another signal
            second_signal = create_signal(
                "Reference", x=[1, 2, 3, 4, 5], y=[2, 3, 4, 5, 6]
            )
            combined = dst_2_to_1(denoised, second_signal, "subtract")
            assert combined.title == "subtract({0}, {1})"

        finally:
            set_default_title_formatter(original_formatter)

    def test_formatter_switching(self):
        """Test switching between formatters (for testing purposes)."""
        original_formatter = get_default_title_formatter()

        try:
            # Start with DataLab's default (PlaceholderTitleFormatter)
            assert isinstance(get_default_title_formatter(), PlaceholderTitleFormatter)

            signal = create_signal("Test", x=[1, 2, 3], y=[4, 5, 6])

            # Test with placeholder formatter
            result1 = dst_1_to_1(signal, "normalize")
            assert result1.title == "normalize({0})"

            # Switch to simple formatter (for comparison)
            set_default_title_formatter(SimpleTitleFormatter())
            result2 = dst_1_to_1(signal, "normalize")
            assert result2.title == "Normalize Result"

            # Switch back to placeholder formatter
            set_default_title_formatter(PlaceholderTitleFormatter())
            result3 = dst_1_to_1(signal, "normalize")
            assert result3.title == "normalize({0})"

        finally:
            set_default_title_formatter(original_formatter)

    def test_error_handling_in_title_formatting(self):
        """Test error handling in title formatting operations."""
        original_formatter = get_default_title_formatter()
        try:
            formatter = PlaceholderTitleFormatter()
            set_default_title_formatter(formatter)

            # Test with empty object list (should raise IndexError)
            with pytest.raises(IndexError):
                formatter.resolve_placeholder_title("process({0})", [])

            # Test with None values (should be handled gracefully)
            signal = create_signal("Test", x=[1, 2], y=[3, 4])
            result = dst_1_to_1(signal, "test_func", None)
            assert result.title == "test_func({0})"

            result = dst_1_to_1(signal, "test_func", "")
            assert result.title == "test_func({0})"

        finally:
            set_default_title_formatter(original_formatter)


class TestDataLabCompatibilityPatterns:
    """Test suite for DataLab-specific compatibility patterns."""

    def test_datalab_object_naming_patterns(self):
        """Test compatibility with DataLab's object naming conventions."""
        import numpy as np

        formatter = PlaceholderTitleFormatter()

        # Create objects that simulate DataLab naming patterns
        signal1 = create_signal("s000001", x=[1, 2], y=[3, 4])
        signal2 = create_signal("s000002", x=[1, 2], y=[5, 6])
        image1 = create_image("i000001", data=np.array([[1, 2], [3, 4]]))

        # Test various DataLab-style operations
        test_cases = [
            # Signal operations
            ("fft({0})", [signal1], "fft(obj1)"),
            (
                "cross_correlation({0}, {1})",
                [signal1, signal2],
                "cross_correlation(obj1, obj2)",
            ),
            (
                "bandpass_filter({0})|f_low=10,f_high=100",
                [signal1],
                "bandpass_filter(obj1)|f_low=10,f_high=100",
            ),
            # Image operations
            ("threshold({0})", [image1], "threshold(obj1)"),
            (
                "morphology({0})|operation=opening",
                [image1],
                "morphology(obj1)|operation=opening",
            ),
        ]

        for placeholder_title, objects, expected in test_cases:
            result = formatter.resolve_placeholder_title(placeholder_title, objects)
            assert result == expected, (
                f"Failed for {placeholder_title}: got {result}, expected {expected}"
            )

    def test_complex_parameter_strings(self):
        """Test handling of complex parameter strings typical in DataLab."""
        original_formatter = get_default_title_formatter()
        try:
            set_default_title_formatter(PlaceholderTitleFormatter())

            signal = create_signal("Test", x=[1, 2, 3], y=[4, 5, 6])

            # Test complex parameter strings
            complex_params = [
                "sigma=1.5,mode=constant",
                "window=hamming,nperseg=256,noverlap=128",
                "method=butterworth,order=4,critical_freq=0.1",
                "roi=[10,20,30,40],background=auto",
            ]

            for params in complex_params:
                result = dst_1_to_1(signal, "filter", params)
                expected = f"filter({{0}})|{params}"
                assert result.title == expected

        finally:
            set_default_title_formatter(original_formatter)


if __name__ == "__main__":
    pytest.main([__file__])
