# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Fix SVG icons by removing conflicting fill attributes.

This script automatically finds and fixes SVG icons that have conflicting
fill attributes (both fill="..." and style="fill:...") which cause
inconsistent rendering between Qt and Windows native renderers.
"""

import re
from pathlib import Path


def has_conflicting_fill(content: str) -> bool:
    """Check if SVG content has conflicting fill attributes.

    Args:
        content: SVG file content

    Returns:
        True if conflicting fill attributes are found
    """
    # Check for elements with both fill attribute and style with fill
    pattern1 = (
        r"<(circle|rect|path|polygon|ellipse)[^>]*"
        r'fill="[^"]*"[^>]*style="[^"]*fill:[^"]+"'
    )
    pattern2 = (
        r"<(circle|rect|path|polygon|ellipse)[^>]*"
        r'style="[^"]*fill:[^"]+"[^>]*fill="[^"]*"'
    )
    return bool(re.search(pattern1, content) or re.search(pattern2, content))


def find_problematic_icons(icons_dir: Path) -> list[Path]:
    """Find all SVG icons with conflicting fill attributes.

    Args:
        icons_dir: Path to the icons directory

    Returns:
        List of paths to problematic SVG files
    """
    problematic = []
    for svg_file in icons_dir.rglob("*.svg"):
        content = svg_file.read_text(encoding="utf-8")
        if has_conflicting_fill(content):
            problematic.append(svg_file)
    return problematic


def fix_icon(filepath: str) -> None:
    """Fix an icon by removing conflicting fill attributes from elements
    that have both fill attribute and style with fill.

    Args:
        filepath: Path to the SVG file
    """
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")

    # Pattern to find elements with both fill="..." and style="...fill:..."
    def fix_element(match):
        element = match.group(0)
        # Check if has both fill attribute and style with fill
        if 'fill="' in element and 'style="' in element:
            # Extract fill color from style
            style_match = re.search(r'style="[^"]*fill:\s*([^;"\s]+)', element)
            if style_match:
                fill_color = style_match.group(1)
                # Remove the conflicting fill attribute
                element = re.sub(r'\s+fill="[^"]*"', "", element)
                # Remove the style attribute
                element = re.sub(r'\s+style="[^"]*"', "", element)
                # Add the fill from style as a direct attribute before the closing
                if element.endswith("/>"):
                    element = element[:-2] + f' fill="{fill_color}" fill-opacity="1" />'
                elif element.endswith(">"):
                    element = element[:-1] + f' fill="{fill_color}" fill-opacity="1">'
        return element

    # Apply to all elements
    for tag in ["circle", "rect", "path", "polygon", "ellipse"]:
        pattern = f"<{tag}[^>]*/?>"
        content = re.sub(pattern, fix_element, content)

    path.write_text(content, encoding="utf-8")
    print(f"Fixed: {filepath}")


def main():
    """Find and fix all icon files with conflicting fill attributes."""
    icons_dir = Path(__file__).parent.parent / "datalab" / "data" / "icons"

    print(f"Scanning {icons_dir} for SVG icons with conflicting fill attributes...")
    problematic_files = find_problematic_icons(icons_dir)

    if not problematic_files:
        print("No icons with conflicting fill attributes found!")
        return

    print(f"Found {len(problematic_files)} icons with conflicting fill attributes:")
    for file in problematic_files:
        rel_path = file.relative_to(icons_dir)
        print(f"  - {rel_path}")

    print("\nFixing icons...")
    for file in problematic_files:
        fix_icon(str(file))

    print(f"\nSuccessfully fixed {len(problematic_files)} icons!")


if __name__ == "__main__":
    main()
