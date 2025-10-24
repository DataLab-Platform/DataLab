# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for Add Metadata feature.

This module tests the `AddMetadataParam` class and the `add_metadata` method
from `datalab.gui.panel.base`.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING

import guidata.config as gcfg
import numpy as np
import pytest
from sigima.objects import create_image, create_signal

from datalab.env import execenv
from datalab.gui.panel.base import AddMetadataParam

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj


def create_test_signals() -> list[SignalObj]:
    """Create a list of test signals for testing."""
    # Signal 1: Sine wave
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    signal1 = create_signal(title="Sine Wave", x=x, y=y)

    # Signal 2: Cosine wave
    y = np.cos(x * 2)
    signal2 = create_signal(title="Cosine Wave", x=x, y=y)

    # Signal 3: Exponential decay
    y = np.exp(-x / 3)
    signal3 = create_signal(title="Exponential Decay", x=x, y=y)

    return [signal1, signal2, signal3]


def create_test_images() -> list[ImageObj]:
    """Create a list of test images for testing."""
    # Image 1: Random noise
    data1 = np.random.rand(100, 100)
    image1 = create_image(title="Random Noise", data=data1)

    # Image 2: Gaussian pattern
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    data2 = np.exp(-(X**2 + Y**2))
    image2 = create_image(title="Gaussian Pattern", data=data2)

    # Image 3: Checkerboard pattern
    data3 = np.zeros((100, 100))
    data3[::10, ::10] = 1
    data3[5::10, 5::10] = 1
    image3 = create_image(title="Checkerboard", data=data3)

    return [image1, image2, image3]


class TestAddMetadata:
    """Test class for AddMetadataParam with proper setup/teardown."""

    @pytest.fixture(autouse=True)
    def setup_validation(self):
        """Disable guidata validation during tests."""
        old_mode = gcfg.get_validation_mode()
        gcfg.set_validation_mode(gcfg.ValidationMode.DISABLED)
        yield
        gcfg.set_validation_mode(old_mode)

    def test_string_values(self) -> None:
        """Test adding string metadata values."""
        execenv.print(f"{self.test_string_values.__doc__}:")

        # Create test signals
        signals = create_test_signals()

        # Create parameter
        p = AddMetadataParam(signals)
        p.metadata_key = "test_string"
        p.value_pattern = "{title}"
        p.conversion = "string"

        # Build values
        values = p.build_values(signals)

        expected_values = ["Sine Wave", "Cosine Wave", "Exponential Decay"]
        assert values == expected_values, f"Expected {expected_values}, got {values}"
        execenv.print(f"  ✓ String values: {values}")

        # Check preview
        p.update_preview()
        assert "test_string" in p.preview, "Preview should contain metadata key"
        execenv.print("  ✓ Preview generated successfully")

        execenv.print(f"{self.test_string_values.__doc__}: OK")

    def test_numeric_values(self) -> None:
        """Test adding numeric metadata values."""
        execenv.print(f"{self.test_numeric_values.__doc__}:")

        # Create test signals
        signals = create_test_signals()

        # Test integer conversion
        p = AddMetadataParam(signals)
        p.metadata_key = "index"
        p.value_pattern = "{index}"
        p.conversion = "int"

        values = p.build_values(signals)
        expected_values = [1, 2, 3]
        assert values == expected_values, f"Expected {expected_values}, got {values}"
        execenv.print(f"  ✓ Integer values: {values}")

        # Test float conversion
        p.conversion = "float"
        values = p.build_values(signals)
        expected_values = [1.0, 2.0, 3.0]
        assert values == expected_values, f"Expected {expected_values}, got {values}"
        execenv.print(f"  ✓ Float values: {values}")

        execenv.print(f"{self.test_numeric_values.__doc__}: OK")

    def test_boolean_values(self) -> None:
        """Test adding boolean metadata values."""
        execenv.print(f"{self.test_boolean_values.__doc__}:")

        # Create test signals
        signals = create_test_signals()

        # Test boolean conversion with "true" pattern
        p = AddMetadataParam(signals)
        p.metadata_key = "is_valid"
        p.value_pattern = "true"
        p.conversion = "bool"

        values = p.build_values(signals)
        expected_values = [True, True, True]
        assert values == expected_values, f"Expected {expected_values}, got {values}"
        execenv.print(f"  ✓ Boolean (true) values: {values}")

        # Test boolean conversion with "false" pattern
        p.value_pattern = "false"
        values = p.build_values(signals)
        expected_values = [False, False, False]
        assert values == expected_values, f"Expected {expected_values}, got {values}"
        execenv.print(f"  ✓ Boolean (false) values: {values}")

        execenv.print(f"{self.test_boolean_values.__doc__}: OK")

    def test_pattern_formatting(self) -> None:
        """Test various pattern formatting options."""
        execenv.print(f"{self.test_pattern_formatting.__doc__}:")

        # Create test signals
        signals = create_test_signals()

        # Test index with zero padding
        p = AddMetadataParam(signals)
        p.metadata_key = "file_id"
        p.value_pattern = "file_{index:03d}"
        p.conversion = "string"

        values = p.build_values(signals)
        expected_values = ["file_001", "file_002", "file_003"]
        assert values == expected_values, f"Expected {expected_values}, got {values}"
        execenv.print(f"  ✓ Padded index pattern: {values}")

        # Test uppercase modifier
        p.value_pattern = "{title:upper}"
        values = p.build_values(signals)
        expected_values = ["SINE WAVE", "COSINE WAVE", "EXPONENTIAL DECAY"]
        assert values == expected_values, f"Expected {expected_values}, got {values}"
        execenv.print(f"  ✓ Uppercase pattern: {values}")

        # Test lowercase modifier
        p.value_pattern = "{title:lower}"
        values = p.build_values(signals)
        expected_values = ["sine wave", "cosine wave", "exponential decay"]
        assert values == expected_values, f"Expected {expected_values}, got {values}"
        execenv.print(f"  ✓ Lowercase pattern: {values}")

        execenv.print(f"{self.test_pattern_formatting.__doc__}: OK")

    def test_with_images(self) -> None:
        """Test AddMetadataParam with image objects."""
        execenv.print(f"{self.test_with_images.__doc__}:")

        # Create test images
        images = create_test_images()

        # Create parameter
        p = AddMetadataParam(images)
        p.metadata_key = "category"
        p.value_pattern = "{title:lower}"
        p.conversion = "string"

        # Build values
        values = p.build_values(images)
        expected_values = ["random noise", "gaussian pattern", "checkerboard"]
        assert values == expected_values, f"Expected {expected_values}, got {values}"
        execenv.print(f"  ✓ Image metadata values: {values}")

        # Check preview
        p.update_preview()
        assert "category" in p.preview, "Preview should contain metadata key"
        # Check that all values are in the preview
        for expected_val in expected_values:
            assert expected_val in p.preview, f"Preview should contain {expected_val}"
        execenv.print("  ✓ Preview contains all values")

        execenv.print(f"{self.test_with_images.__doc__}: OK")

    def test_error_handling(self) -> None:
        """Test error handling for invalid patterns."""
        execenv.print(f"{self.test_error_handling.__doc__}:")

        # Create test signals
        signals = create_test_signals()

        # Test with invalid pattern
        p = AddMetadataParam(signals)
        p.metadata_key = "test"
        p.value_pattern = "{invalid_key}"  # This key doesn't exist
        p.conversion = "string"

        # Update preview should handle the error gracefully
        p.update_preview()
        assert "Invalid" in p.preview, "Preview should show invalid pattern error"
        execenv.print("  ✓ Invalid pattern error handled")

        execenv.print(f"{self.test_error_handling.__doc__}: OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
