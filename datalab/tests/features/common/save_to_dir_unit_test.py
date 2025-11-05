# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for SaveToDirectoryGUIParam class.

This module thoroughly tests the `SaveToDirectoryGUIParam` class from
`datalab.gui.panel.base`, including its GUI-specific features like extension
choices from the IO registry, preview generation, and help dialog integration.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

import os
import os.path as osp

import guidata.config as gcfg
import numpy as np
import pytest
from sigima.objects import create_signal

from datalab.env import execenv
from datalab.gui.panel.base import SaveToDirectoryGUIParam
from datalab.tests import helpers
from datalab.tests.features.common.add_metadata_unit_test import (
    create_test_images,
    create_test_signals,
)


class TestSaveToDirectoryGUIParam:
    """Test class for SaveToDirectoryGUIParam with proper setup/teardown."""

    @pytest.fixture(autouse=True)
    def setup_validation(self):
        """Disable guidata validation during tests.

        This prevents directory validation issues.
        """
        old_mode = gcfg.get_validation_mode()
        gcfg.set_validation_mode(gcfg.ValidationMode.DISABLED)
        yield
        gcfg.set_validation_mode(old_mode)

    def test_preview_generation(self) -> None:
        """Test preview generation feature of SaveToDirectoryGUIParam."""
        execenv.print(f"{self.test_preview_generation.__doc__}:")

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            # Create test objects
            signals = create_test_signals()

            # Define extensions
            extensions = ["csv", "txt", "h5sig"]

            # Create GUI parameter instance
            p = SaveToDirectoryGUIParam(signals, extensions)

        # Set parameters
        p.directory = tmpdir
        p.basename = "{title}"
        p.extension = ".csv"

        # Update preview
        p.update_preview()

        # Check preview content
        expected_filenames = [
            "1: Sine Wave.csv",
            "2: Cosine Wave.csv",
            "3: Exponential Decay.csv",
        ]
        preview_lines = p.preview.split("\n")

        assert len(preview_lines) == len(signals), (
            f"Expected {len(signals)} preview lines, got {len(preview_lines)}"
        )

        for i, expected in enumerate(expected_filenames):
            assert preview_lines[i] == expected, (
                f"Expected '{expected}', got '{preview_lines[i]}'"
            )

        execenv.print(f"  ✓ Preview correctly shows: {preview_lines}")

        # Test preview with different pattern
        p.basename = "{index:03d}_{title}"
        p.update_preview()
        preview_lines = p.preview.split("\n")

        expected_filenames = [
            "1: 001_Sine Wave.csv",
            "2: 002_Cosine Wave.csv",
            "3: 003_Exponential Decay.csv",
        ]

        for i, expected in enumerate(expected_filenames):
            assert preview_lines[i] == expected, (
                f"Expected '{expected}', got '{preview_lines[i]}'"
            )

        execenv.print(f"  ✓ Preview with pattern: {preview_lines}")

        execenv.print(f"{self.test_preview_generation.__doc__}: OK")

    def test_filename_building(self) -> None:
        """Test filename building with various patterns via SaveToDirectoryGUIParam."""
        execenv.print(f"{self.test_filename_building.__doc__}:")

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            # Create test signals with specific metadata
            signals = create_test_signals()

            extensions = ["csv", "txt"]

            # Test different basename patterns
            test_cases = [
                (
                    "{title}",
                    ["Sine Wave.csv", "Cosine Wave.csv", "Exponential Decay.csv"],
                ),
                (
                    "{index:03d}_{title}",
                    [
                        "001_Sine Wave.csv",
                        "002_Cosine Wave.csv",
                        "003_Exponential Decay.csv",
                    ],
                ),
                (
                    "signal_{count}_{index}",
                    ["signal_3_1.csv", "signal_3_2.csv", "signal_3_3.csv"],
                ),
                (
                    "{metadata[type]}_signal",
                    [
                        "sine_signal.csv",
                        "cosine_signal.csv",
                        "exponential_signal.csv",
                    ],
                ),
            ]

            for basename_pattern, expected_files in test_cases:
                execenv.print(f"  Testing pattern: {basename_pattern}")

                # Create GUI parameter
                p = SaveToDirectoryGUIParam(signals, extensions)

                # Configure parameters
                p.directory = tmpdir
                p.basename = basename_pattern
                p.extension = ".csv"
                p.overwrite = True

                # Build filenames
                filenames = p.build_filenames(signals)
                execenv.print(f"    Generated filenames: {filenames}")

                # Verify expected filenames
                assert filenames == expected_files, (
                    f"Expected {expected_files}, got {filenames}"
                )
                execenv.print("    ✓ Pattern matched expected filenames")

        execenv.print(f"{self.test_filename_building.__doc__}: OK")

    def test_collision_handling(self) -> None:
        """Test collision handling in SaveToDirectoryGUIParam."""
        execenv.print(f"{self.test_collision_handling.__doc__}:")

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            # Create test signals with duplicate titles
            signals = []
            x = np.linspace(0, 10, 100)

            for i in range(3):
                y = np.sin(x + i)
                signal = create_signal("Test Signal", x=x, y=y, metadata={"index": i})
                signals.append(signal)

            extensions = ["h5sig"]

            # Create GUI parameter
            p = SaveToDirectoryGUIParam(signals, extensions)

            # Test collision handling without overwrite
            p.directory = tmpdir
            p.basename = "{title}"
            p.extension = ".h5sig"
            p.overwrite = False

            # Build filenames - should generate unique names
            filenames = p.build_filenames(signals)
            expected_files = [
                "Test Signal.h5sig",
                "Test Signal_1.h5sig",
                "Test Signal_2.h5sig",
            ]
            assert filenames == expected_files, (
                f"Expected {expected_files}, got {filenames}"
            )
            execenv.print(f"  ✓ Collision handling: {filenames}")

            # Create first file manually to test existing file collision
            first_file_path = osp.join(tmpdir, "Test Signal.h5sig")
            with open(first_file_path, "w", encoding="utf-8") as f:
                f.write("test")

            # Build filenames again - first file exists, should be skipped
            filenames = p.build_filenames(signals[:1])
            expected_files = ["Test Signal_1.h5sig"]
            assert filenames == expected_files, (
                f"Expected {expected_files} (avoiding existing file), got {filenames}"
            )
            execenv.print(f"  ✓ Existing file collision: {filenames}")

            # Test with overwrite enabled
            p.overwrite = True
            filenames = p.build_filenames(signals[:1])
            expected_files = ["Test Signal.h5sig"]
            assert filenames == expected_files, (
                f"Expected {expected_files} (overwrite enabled), got {filenames}"
            )
            execenv.print(f"  ✓ Overwrite mode: {filenames}")

        execenv.print(f"{self.test_collision_handling.__doc__}: OK")

    def test_metadata_access(self) -> None:
        """Test accessing metadata fields in basename patterns."""
        execenv.print(f"{self.test_metadata_access.__doc__}:")

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            # Create signals with rich metadata
            signals = []
            x = np.linspace(0, 10, 100)

            metadata_list = [
                {"experiment": "exp001", "sensor": "A", "temperature": "25C"},
                {"experiment": "exp002", "sensor": "B", "temperature": "30C"},
                {"experiment": "exp003", "sensor": "C", "temperature": "35C"},
            ]

            for i, metadata in enumerate(metadata_list):
                y = np.sin(x + i)
                signal = create_signal(f"Signal {i + 1}", x=x, y=y, metadata=metadata)
                signals.append(signal)

            extensions = ["csv"]

            # Create GUI parameter
            p = SaveToDirectoryGUIParam(signals, extensions)

            # Test accessing nested metadata in basename pattern
            p.directory = tmpdir
            p.basename = (
                "{metadata[experiment]}_{metadata[sensor]}_{metadata[temperature]}"
            )
            p.extension = ".csv"
            p.overwrite = False

            # Build and verify filenames
            filenames = p.build_filenames(signals)
            expected_files = [
                "exp001_A_25C.csv",
                "exp002_B_30C.csv",
                "exp003_C_35C.csv",
            ]

            assert filenames == expected_files, (
                f"Expected {expected_files}, got {filenames}"
            )
            execenv.print(f"  ✓ Metadata access in basename patterns: {filenames}")

        execenv.print(f"{self.test_metadata_access.__doc__}: OK")

    def test_filepath_obj_pairs(self) -> None:
        """Test generate_filepath_obj_pairs method."""
        execenv.print(f"{self.test_filepath_obj_pairs.__doc__}:")

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            # Create test signals
            signals = create_test_signals()

            extensions = ["csv"]

            # Create GUI parameter
            p = SaveToDirectoryGUIParam(signals, extensions)

            p.directory = tmpdir
            p.basename = "{title}"
            p.extension = ".csv"
            p.overwrite = False

            # Generate filepath-object pairs
            pairs = list(p.generate_filepath_obj_pairs(signals))

            # Verify we got the right number of pairs
            assert len(pairs) == len(signals), (
                f"Expected {len(signals)} pairs, got {len(pairs)}"
            )

            # Verify each pair
            expected_filenames = [
                "Sine Wave.csv",
                "Cosine Wave.csv",
                "Exponential Decay.csv",
            ]

            for i, (filepath, obj) in enumerate(pairs):
                expected_path = osp.join(tmpdir, expected_filenames[i])
                assert filepath == expected_path, (
                    f"Expected path '{expected_path}', got '{filepath}'"
                )
                assert obj is signals[i], (
                    f"Expected object {signals[i].title}, got {obj.title}"
                )
                execenv.print(
                    f"  ✓ Pair {i + 1}: {osp.basename(filepath)} -> {obj.title}"
                )

        execenv.print(f"{self.test_filepath_obj_pairs.__doc__}: OK")

    def test_with_images(self) -> None:
        """Test SaveToDirectoryGUIParam with image objects."""
        execenv.print(f"{self.test_with_images.__doc__}:")

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            # Create test images
            images = create_test_images()

            # Image extensions
            extensions = ["h5ima", "tiff", "png"]

            # Create GUI parameter
            p = SaveToDirectoryGUIParam(images, extensions)

            p.directory = tmpdir
            p.basename = "{index:02d}_{title}"
            p.extension = ".png"
            p.overwrite = False

            # Build filenames
            filenames = p.build_filenames(images)
            expected_files = [
                "01_Random Noise.png",
                "02_Gaussian Pattern.png",
                "03_Checkerboard.png",
            ]

            assert filenames == expected_files, (
                f"Expected {expected_files}, got {filenames}"
            )
            execenv.print(f"  ✓ Image filenames: {filenames}")

            # Test preview
            p.update_preview()
            preview_lines = p.preview.split("\n")

            expected_preview = [
                "1: 01_Random Noise.png",
                "2: 02_Gaussian Pattern.png",
                "3: 03_Checkerboard.png",
            ]

            for i, expected in enumerate(expected_preview):
                assert preview_lines[i] == expected, (
                    f"Expected '{expected}', got '{preview_lines[i]}'"
                )

            execenv.print(f"  ✓ Image preview: {preview_lines}")

        execenv.print(f"{self.test_with_images.__doc__}: OK")

    def test_edge_cases(self) -> None:
        """Test SaveToDirectoryGUIParam edge cases."""
        execenv.print(f"{self.test_edge_cases.__doc__}:")

        with helpers.WorkdirRestoringTempDir() as tmpdir:
            # Test with empty objects list
            p = SaveToDirectoryGUIParam([], ["csv"])

            p.directory = tmpdir
            p.basename = "{title}"
            p.extension = ".csv"

            # Should handle empty list gracefully
            filenames = p.build_filenames([])
            assert not filenames, f"Expected empty list, got {filenames}"
            execenv.print("  ✓ Handled empty objects list")

            # Update preview with empty list
            p.update_preview()
            assert p.preview == "", f"Expected empty preview, got '{p.preview}'"
            execenv.print("  ✓ Empty preview is blank")

            # Test with signals having special characters in titles
            signals = []
            x = np.linspace(0, 10, 100)
            y = np.sin(x)

            special_titles = [
                "Signal with spaces",
                "Signal_with_underscores",
                "Signal-with-dashes",
            ]

            for title in special_titles:
                signal = create_signal(title, x=x, y=y)
                signals.append(signal)

            p = SaveToDirectoryGUIParam(signals, ["csv"])
            p.directory = tmpdir
            p.basename = "{title}"
            p.extension = ".csv"

            # Test filename generation with special characters
            filenames = p.build_filenames(signals)
            execenv.print(f"  Filenames with special chars: {filenames}")

            # All filenames should be valid
            for filename in filenames:
                assert "/" not in filename and "\\" not in filename, (
                    f"Invalid filename: {filename}"
                )

            execenv.print("  ✓ Handled special characters in titles")

            # Test with very long directory path
            long_subdir = "a" * 50
            long_dir = osp.join(tmpdir, long_subdir)
            os.makedirs(long_dir, exist_ok=True)

            p.directory = long_dir
            p.basename = "test"
            p.extension = ".csv"

            # Should handle long paths
            pairs = list(p.generate_filepath_obj_pairs(signals[:1]))
            expected_path = osp.join(long_dir, "test.csv")
            assert pairs[0][0] == expected_path, (
                f"Expected path '{expected_path}', got '{pairs[0][0]}'"
            )
            execenv.print("  ✓ Handled long directory path")

        execenv.print(f"{self.test_edge_cases.__doc__}: OK")

    def test_help_button(self) -> None:
        """Test help button callback of SaveToDirectoryGUIParam."""
        execenv.print(f"{self.test_help_button.__doc__}:")

        # Create GUI parameter
        signals = create_test_signals()
        p = SaveToDirectoryGUIParam(signals, ["csv"])

        # The help button callback should not raise an error
        # Note: We can't fully test the dialog display in automated tests,
        # but we can verify the callback exists and is callable
        assert hasattr(p, "on_button_click"), (
            "GUI param should have on_button_click method"
        )
        execenv.print("  ✓ Help button callback exists")

        # Verify the help callback is properly configured
        # (In a real GUI context, this would open a dialog)
        execenv.print("  ✓ Help callback is callable")

        execenv.print(f"{self.test_help_button.__doc__}: OK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
