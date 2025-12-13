#!/usr/bin/env python3
"""Script to identify which SVG files trigger Qt warnings.

This script scans all SVG files in the icon directories and loads them
one by one, reporting which files cause Qt SVG warnings.
"""

from __future__ import annotations

import sys
from pathlib import Path

from guidata.qthelpers import qt_app_context
from qtpy import QtCore as QC
from qtpy import QtGui as QG

from datalab.config import DATAPATH

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Track which file is currently being loaded
current_file: Path | None = None
problematic_files: set[Path] = set()


def message_handler(
    msg_type: QC.QtMsgType, context: QC.QMessageLogContext, message: str
) -> None:
    """Custom Qt message handler to catch SVG warnings."""
    if "svg" in message.lower() or "path" in message.lower():
        if current_file is not None and current_file not in problematic_files:
            problematic_files.add(current_file)
            print(f"  ⚠️  {current_file.name}: {message}")


def scan_svg_directory(svg_dir: Path) -> None:
    """Scan all SVG files in a directory and test loading them."""
    if not svg_dir.exists():
        print(f"Directory not found: {svg_dir}")
        return

    svg_files = sorted(svg_dir.rglob("*.svg"))
    print(f"\nScanning {len(svg_files)} SVG files in {svg_dir}...")
    print("-" * 60)

    global current_file
    for svg_file in svg_files:
        current_file = svg_file
        relative_path = svg_file.relative_to(svg_dir)
        # Load the icon - this triggers Qt SVG parsing
        icon = QG.QIcon(str(svg_file))
        # Force the icon to actually render by getting a pixmap
        pixmap = icon.pixmap(32, 32)
        if pixmap.isNull():
            print(f"  ❌ Failed to load: {relative_path}")

    current_file = None


def main() -> None:
    """Main function to scan SVG icons."""
    with qt_app_context():
        # Install custom message handler
        QC.qInstallMessageHandler(message_handler)

        print("=" * 60)
        print("SVG Icon Scanner - Detecting problematic SVG files")
        print("=" * 60)

        # Scan DataLab icons
        scan_svg_directory(Path(DATAPATH))

        # Try to find and scan guidata icons
        try:
            import guidata

            guidata_path = Path(guidata.__file__).parent
            guidata_icons = guidata_path / "data" / "icons"
            scan_svg_directory(guidata_icons)
        except ImportError:
            print("\nguidata not found, skipping its icons")

        # Try to find and scan plotpy icons
        try:
            import plotpy

            plotpy_path = Path(plotpy.__file__).parent
            plotpy_icons = plotpy_path / "data" / "icons"
            scan_svg_directory(plotpy_icons)
        except ImportError:
            print("\nplotpy not found, skipping its icons")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if problematic_files:
            print(f"\n❌ Found {len(problematic_files)} problematic SVG file(s):\n")
            for file_path in sorted(problematic_files):
                print(f"  • {file_path}\n")
        else:
            print("\n✅ No problematic SVG files found!")


if __name__ == "__main__":
    main()
