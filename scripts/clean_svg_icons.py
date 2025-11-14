#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean SVG icons by removing Inkscape metadata and fixing conflicting attributes.

This script:
1. Removes Inkscape/Sodipodi namespaces and elements
2. Fixes conflicting fill/stroke attributes (removes when in conflict with style)
3. Simplifies style attributes to presentation attributes
4. Removes unnecessary IDs
"""

import re
from pathlib import Path


def clean_svg_content(content: str) -> tuple[str, bool]:
    """Clean SVG content by removing Inkscape metadata and fixing conflicts.

    Args:
        content: Original SVG content

    Returns:
        Tuple of (cleaned_content, was_modified)
    """
    original = content
    modified = False

    # Remove Inkscape/Sodipodi namespace declarations from svg tag
    content = re.sub(r'\s+xmlns:inkscape="[^"]*"', "", content)
    content = re.sub(r'\s+xmlns:sodipodi="[^"]*"', "", content)
    content = re.sub(r'\s+xmlns:svg="[^"]*"', "", content)

    # Remove inkscape/sodipodi attributes from svg tag
    content = re.sub(r'\s+(?:inkscape|sodipodi):[a-zA-Z-]+="[^"]*"', "", content)

    # Remove entire sodipodi:namedview elements
    content = re.sub(
        r"<sodipodi:namedview[^>]*>.*?</sodipodi:namedview>",
        "",
        content,
        flags=re.DOTALL,
    )

    # Remove self-closing sodipodi:namedview elements
    content = re.sub(r"<sodipodi:namedview[^>]*/>\s*", "", content)

    # Remove empty defs elements
    content = re.sub(r'<defs\s+id="[^"]*"\s*/>\s*', "", content)

    # Remove inkscape/sodipodi attributes from other elements
    content = re.sub(r'\s+(?:inkscape|sodipodi):[a-zA-Z-]+="[^"]*"', "", content)

    # Fix conflicting fill attributes: remove fill="none" when style has fill
    # Pattern: find elements with both fill="none" and style="...fill:..."
    def fix_fill_conflict(match):
        element = match.group(0)
        # If has style with fill, remove fill="none"
        if "style=" in element and re.search(r'fill:[^;"]+', element):
            element = re.sub(r'\s+fill="none"', "", element)
        return element

    # Apply to circle, rect, path, polygon, ellipse, etc.
    for tag in ["circle", "rect", "path", "polygon", "ellipse", "line", "polyline"]:
        pattern = f"<{tag}[^>]*>"
        content = re.sub(pattern, fix_fill_conflict, content)

    # Fix conflicting stroke attributes: remove stroke="..." when style has stroke
    def fix_stroke_conflict(match):
        element = match.group(0)
        # If has style with stroke, remove conflicting stroke attribute
        if "style=" in element and re.search(r'stroke:[^;"]+', element):
            # Only remove if there's a direct stroke attribute that conflicts
            if re.search(r'\s+stroke="[^"]*"', element):
                # Keep the style version, remove the direct attribute
                element = re.sub(r'\s+stroke="[^"]*"', "", element)
        return element

    for tag in ["circle", "rect", "path", "polygon", "ellipse", "line", "polyline"]:
        pattern = f"<{tag}[^>]*>"
        content = re.sub(pattern, fix_stroke_conflict, content)

    # Convert style attributes to presentation attributes for better compatibility
    def style_to_attrs(match):
        element = match.group(0)

        # Extract style content
        style_match = re.search(r'style="([^"]*)"', element)
        if not style_match:
            return element

        style_content = style_match.group(1)

        # Parse style properties
        attrs_to_add = []
        for prop in style_content.split(";"):
            prop = prop.strip()
            if not prop:
                continue
            if ":" in prop:
                key, value = prop.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Convert common properties
                if key in [
                    "fill",
                    "fill-opacity",
                    "stroke",
                    "stroke-width",
                    "stroke-opacity",
                    "stroke-dasharray",
                    "opacity",
                ]:
                    # Check if attribute already exists
                    if not re.search(rf"\s+{re.escape(key)}=", element):
                        attrs_to_add.append(f'{key}="{value}"')

        if attrs_to_add:
            # Remove the style attribute
            element = re.sub(r'\s+style="[^"]*"', "", element)
            # Add new attributes before the closing >
            element = element.rstrip(">").rstrip("/") + " " + " ".join(attrs_to_add)
            if match.group(0).endswith("/>"):
                element += " />"
            else:
                element += ">"

        return element

    # Apply to all shape elements
    for tag in ["circle", "rect", "path", "polygon", "ellipse", "line", "polyline"]:
        pattern = f"(<{tag}[^>]*>)"
        content = re.sub(pattern, style_to_attrs, content)

    # Clean up multiple spaces
    content = re.sub(r"  +", " ", content)

    # Clean up space before />
    content = re.sub(r"\s+/>", " />", content)

    # Improve formatting
    content = re.sub(r">\s*<", ">\n  <", content)
    content = re.sub(r"(<svg[^>]*>)\s*", r"\1\n  ", content)
    content = re.sub(r"\s*(</svg>)", r"\n\1", content)

    # Check if modified
    modified = content != original

    return content, modified


def clean_svg_file(filepath: Path) -> bool:
    """Clean a single SVG file.

    Args:
        filepath: Path to SVG file

    Returns:
        True if file was modified
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            original = f.read()

        cleaned, modified = clean_svg_content(original)

        if modified:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(cleaned)
            return True
        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Clean all SVG files in the icons directory."""
    icons_dir = Path(__file__).parent.parent / "datalab" / "data" / "icons"

    if not icons_dir.exists():
        print(f"Icons directory not found: {icons_dir}")
        return

    svg_files = list(icons_dir.rglob("*.svg"))
    print(f"Found {len(svg_files)} SVG files")

    modified_count = 0
    for svg_file in svg_files:
        if clean_svg_file(svg_file):
            modified_count += 1
            print(f"Cleaned: {svg_file.relative_to(icons_dir)}")

    print(f"\nCleaned {modified_count} out of {len(svg_files)} files")


if __name__ == "__main__":
    main()
