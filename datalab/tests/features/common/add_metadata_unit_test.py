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
from datalab.gui.panel.base import AddMetadataParam, collect_metadata_keys

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj


def create_test_signals() -> list[SignalObj]:
    """Create a list of test signals for testing."""
    # Signal 1: Sine wave
    x = np.linspace(0, 10, 100)
    signal1 = create_signal(
        title="Sine Wave",
        x=x,
        y=np.sin(x),
        metadata={"type": "sine", "frequency": "1 Hz"},
    )

    # Signal 2: Cosine wave
    signal2 = create_signal(
        title="Cosine Wave",
        x=x,
        y=np.cos(x * 2),
        metadata={"type": "cosine", "frequency": "2 Hz"},
    )

    # Signal 3: Exponential decay
    signal3 = create_signal(
        title="Exponential Decay",
        x=x,
        y=np.exp(-x / 3),
        metadata={"type": "exponential", "time_constant": "3 s"},
    )

    return [signal1, signal2, signal3]


def create_test_images() -> list[ImageObj]:
    """Create a list of test images for testing."""
    # Image 1: Random noise
    image1 = create_image(
        title="Random Noise",
        data=np.random.rand(100, 100),
        metadata={"type": "noise", "distribution": "uniform"},
    )

    # Image 2: Gaussian pattern
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    image2 = create_image(
        title="Gaussian Pattern",
        data=np.exp(-(X**2 + Y**2)),
        metadata={"type": "gaussian", "sigma": "1.0"},
    )

    # Image 3: Checkerboard pattern
    data3 = np.zeros((100, 100))
    data3[::10, ::10] = 1
    data3[5::10, 5::10] = 1
    image3 = create_image(
        title="Checkerboard",
        data=data3,
        metadata={"type": "pattern", "period": "10 px"},
    )

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

    def test_namespaced_key(self) -> None:
        """Test that plugin-style namespaced keys are accepted."""
        p = AddMetadataParam()
        item = AddMetadataParam.metadata_key
        for key in ("plugin.org.example.my-plugin.exposure_time_s", "custom_key"):
            assert item.check_value(key), key
        for key in ("1st_key", "bad key", ".hidden"):
            assert not item.check_value(key), key
        del p

    def test_extraction_with_scale(self) -> None:
        """Test extracting a numeric value from titles, with a scale factor."""
        frames = [
            create_signal(title=title, x=np.arange(3.0), y=np.zeros(3))
            for title in ("Flat 5 ms 01", "Dark 01", "Flat 40ms 02")
        ]
        p = AddMetadataParam(frames)
        p.metadata_key = "plugin.org.example.camera.exposure_time_s"
        p.value_pattern = "{title}"
        p.extraction_pattern = r"([\d.]+)\s*ms"
        p.conversion = "float"
        p.scale = 0.001

        assert p.build_values() == [0.005, None, 0.04]
        p.update_preview()
        assert "unchanged" in p.preview and "0.005" in p.preview

        p.if_no_match = "error"
        with pytest.raises(ValueError, match="No match"):
            p.build_values()
        p.update_preview()
        assert p.preview.startswith("Invalid")

    def test_extraction_without_group_and_integer_scale(self) -> None:
        """Test whole-match extraction, integer scaling and invalid patterns."""
        shots = [
            create_signal(title=title, x=np.arange(3.0), y=np.zeros(3))
            for title in ("CH1 shot 042", "CH2 shot 7")
        ]
        p = AddMetadataParam(shots)
        p.value_pattern = "{title}"
        p.extraction_pattern = r"CH\d"
        p.conversion = "string"
        assert p.build_values() == ["CH1", "CH2"]

        p.extraction_pattern = r"shot\s*(\d+)"
        p.conversion = "int"
        assert p.build_values() == [42, 7]
        p.scale = 10.0
        assert p.build_values() == [420, 70]
        p.scale = 0.5
        with pytest.raises(ValueError, match="not an integer"):
            p.build_values()

        p.extraction_pattern = "shot("
        with pytest.raises(ValueError, match="Invalid extraction pattern"):
            p.build_values()

    def test_known_keys(self) -> None:
        """Test suggested keys and their copy into the metadata key."""
        signals = create_test_signals()
        signals[0].metadata["__internal"] = 1
        signals[0].metadata["array"] = np.arange(3)
        known = collect_metadata_keys(signals)
        assert [key for key, _description in known] == [
            "frequency",
            "time_constant",
            "type",
        ]

        p = AddMetadataParam(signals, known)
        choices = [choice[0] for choice in p.get_known_key_choices()]
        assert choices == ["", "frequency", "time_constant", "type"]
        p.on_known_key_changed(None, "frequency")
        assert p.metadata_key == "frequency"
        assert "frequency" in p.preview


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
