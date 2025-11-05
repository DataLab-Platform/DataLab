# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
String utilities unit tests for DataLab

Testing the string utility functions in datalab.utils.strings module,
particularly the save_html_diff function which is useful for debugging
and comparing text differences.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from datalab.utils.strings import save_html_diff


@pytest.fixture
def temp_html_file():
    """Fixture providing a temporary file path for HTML diff output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_diff.html")


def test_save_html_diff_basic(temp_html_file):  # pylint: disable=W0621
    """Test that save_html_diff creates a valid HTML file with diff content."""
    text1 = "Line 1\nLine 2\nLine 3"
    text2 = "Line 1\nModified Line 2\nLine 3"

    with patch("webbrowser.open"):
        save_html_diff(text1, text2, "Before", "After", temp_html_file)

    # Verify file creation and basic structure
    assert Path(temp_html_file).exists()
    content = Path(temp_html_file).read_text(encoding="utf-8")
    assert "<!DOCTYPE html" in content or "<html" in content
    assert "Before" in content
    assert "After" in content


def test_save_html_diff_edge_cases(temp_html_file):  # pylint: disable=W0621
    """Test save_html_diff with edge cases: empty strings and Unicode."""
    # Test with empty string
    with patch("webbrowser.open"):
        save_html_diff("", "New content", "Empty", "Added", temp_html_file)

    assert Path(temp_html_file).exists()

    # Test with Unicode
    with patch("webbrowser.open"):
        save_html_diff(
            "Hello 世界", "Bonjour monde", "中文", "Français", temp_html_file
        )

    content = Path(temp_html_file).read_text(encoding="utf-8")
    assert "中文" in content or "Fran" in content  # Descriptions should be present


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
