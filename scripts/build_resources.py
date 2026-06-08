# -*- coding: utf-8 -*-
"""
Regenerate committed binary graphics resources from their SVG sources.

This script produces the rasterised assets that are checked into git so
that the release pipeline does not need any external rasteriser:

    * ``resources/DataLab.ico``         - multi-size Windows icon for the EXE
    * ``resources/DataLab-Reset.ico``   - multi-size icon for the reset shortcut
    * ``wix/dialog.bmp``                - 493x312 background of the WiX UI dialog
    * ``wix/banner.bmp``                - 493x58 banner used by the WiX UI
    * ``datalab/data/logo/DataLab.svg`` - copy of the master logo (used at runtime)
    * PNG renderings under ``doc/`` and ``datalab/data/logo/`` (splash,
      watermark, banners, frontpage, overview, screenshots, ...)

Run this script only when the corresponding SVG sources change.

Requirements
------------

* Qt (via ``qtpy`` / PyQt5) - already a hard dependency of DataLab, used
  here through ``QSvgRenderer`` for SVG -> raster conversion of the icon
  (multi-size ``.ico``) and WiX bitmap assets.
* CairoSVG - already pulled in by ``sphinxcontrib-svg2pdfconverter[CairoSVG]``
  (PDF documentation build). Used for PNG rasterisation because Qt5's
  ``QSvgRenderer`` mis-renders the Inkscape-authored logos (content drawn
  outside the declared ``width``/``height``, nested transforms): output is
  scaled by an unexpected 4/3 factor and clips the text.
* Pillow - already pulled transitively by scikit-image and PlotPy.
  Handles ICO assembly (with PNG-compressed 256x256 sub-image) and
  24 bpp BMP v3 encoding for the WiX UI assets.

No external tool (Inkscape, ImageMagick, ...) is required.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import shutil
import struct
import sys
from pathlib import Path

import cairosvg
from PIL import Image
from qtpy.QtSvg import QSvgRenderer

REPO_ROOT = Path(__file__).resolve().parent.parent
RESOURCES = REPO_ROOT / "resources"
WIX = REPO_ROOT / "wix"
LOGO = REPO_ROOT / "datalab" / "data" / "logo"
DOC_STATIC = REPO_ROOT / "doc" / "_static"
DOC_IMAGES = REPO_ROOT / "doc" / "images"

ICO_SIZES = (16, 24, 32, 48, 128, 256)
DIALOG_SIZE = (493, 312)
BANNER_SIZE = (493, 58)

# Multi-size ``.ico`` files: (source SVG, destination ICO).
ICO_TARGETS: tuple[tuple[Path, Path], ...] = (
    (RESOURCES / "DataLab.svg", RESOURCES / "DataLab.ico"),
    (RESOURCES / "DataLab-Reset.svg", RESOURCES / "DataLab-Reset.ico"),
)

# PNG renderings: (source SVG, destination PNG, width).
# Height is derived from the SVG ``viewBox`` to preserve the aspect ratio.
PNG_TARGETS: tuple[tuple[Path, Path, int], ...] = (
    (RESOURCES / "DataLab-Title.svg", DOC_STATIC / "DataLab-Title.png", 190),
    (RESOURCES / "DataLab-Frontpage.svg", DOC_STATIC / "DataLab-Frontpage.png", 1300),
    (RESOURCES / "DataLab-Splash.svg", LOGO / "DataLab-Splash.png", 350),
    (RESOURCES / "DataLab-watermark.svg", LOGO / "DataLab-watermark.png", 225),
    (RESOURCES / "DataLab-Banner.svg", DOC_IMAGES / "DataLab-banner.png", 364),
    (RESOURCES / "DataLab-Banner.svg", LOGO / "DataLab-Banner-150.png", 150),
    (
        RESOURCES / "DataLab-Screenshot-Theme.svg",
        DOC_IMAGES / "DataLab-Screenshot-Theme.png",
        982,
    ),
    (RESOURCES / "DataLab-Overview.svg", DOC_IMAGES / "DataLab-Overview.png", 1250),
    (
        RESOURCES / "DataLab-Windows-Installer.svg",
        DOC_IMAGES / "shots" / "windows_installer.png",
        900,
    ),
)

# Plain SVG copies (source -> destination).
SVG_COPIES: tuple[tuple[Path, Path], ...] = (
    (RESOURCES / "DataLab.svg", LOGO / "DataLab.svg"),
)


def _render_svg(src: Path, width: int, height: int):
    """Rasterise ``src`` (SVG) at the requested size and return a PIL RGBA image.

    Uses Qt's ``QSvgRenderer``. Suitable for icon and WiX bitmap assets
    where Qt rendering is known to match the committed reference output.
    For PNG documentation assets, see :func:`_render_svg_png_bytes`
    (cairosvg-based), since Qt5 mis-renders the Inkscape-authored logos.
    """
    from PIL import Image
    from qtpy.QtCore import QSize
    from qtpy.QtGui import QImage, QPainter
    from qtpy.QtSvg import QSvgRenderer

    if not src.is_file():
        raise FileNotFoundError(src)
    renderer = QSvgRenderer(str(src))
    if not renderer.isValid():
        raise RuntimeError(f"Invalid SVG: {src}")
    img = QImage(QSize(width, height), QImage.Format_ARGB32)
    img.fill(0x00000000)  # transparent
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    painter.setRenderHint(QPainter.TextAntialiasing)
    renderer.render(painter)
    painter.end()
    # Convert QImage -> PIL Image (RGBA, top-down).
    img = img.convertToFormat(QImage.Format_RGBA8888)
    ptr = img.constBits()
    try:
        ptr.setsize(img.sizeInBytes())
    except AttributeError:
        # PySide returns a memoryview-like object that already has the right size.
        pass
    return Image.frombytes("RGBA", (width, height), bytes(ptr))


def _svg_height_for_width(src: Path, width: int) -> int:
    """Return the height that preserves the SVG ``viewBox`` aspect ratio."""
    if not src.is_file():
        raise FileNotFoundError(src)
    renderer = QSvgRenderer(str(src))
    if not renderer.isValid():
        raise RuntimeError(f"Invalid SVG: {src}")
    vbox = renderer.viewBoxF()
    if vbox.width() <= 0 or vbox.height() <= 0:
        raise RuntimeError(f"SVG has no usable viewBox: {src}")
    return max(1, round(width * vbox.height() / vbox.width()))


def _build_one_ico(src: Path, dst: Path) -> None:
    """Generate a multi-size ``.ico`` from a single SVG source.

    Each size is rasterised individually via Qt, then assembled into a
    multi-image ``.ico``. The 256x256 sub-image is stored PNG-compressed
    and the smaller sizes as 32-bit BMP (DIB), matching the byte layout
    of the icon originally committed in 2023 and keeping the file
    ~100 KB instead of ~350 KB.
    """
    entries: list[tuple[int, bytes]] = []  # (size, payload bytes as embedded)
    for size in ICO_SIZES:
        img = _render_svg(src, size, size)
        if size >= 256:
            # Store as PNG.
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            entries.append((size, buf.getvalue()))
        else:
            # Store as 32bpp BMP/DIB: BITMAPINFOHEADER + BGRA pixels
            # (bottom-up) + AND mask (all zero, RGBA already has alpha).
            bgra = bytearray(size * size * 4)
            pixels = img.tobytes()  # RGBA, top-down
            row = size * 4
            for y in range(size):
                src_off = (size - 1 - y) * row
                dst_off = y * row
                for x in range(size):
                    r, g, b, a = pixels[src_off + x * 4 : src_off + x * 4 + 4]
                    bgra[dst_off + x * 4 : dst_off + x * 4 + 4] = bytes((b, g, r, a))
            # AND mask: 1 bpp, rows padded to 4 bytes, height = size.
            mask_row = ((size + 31) // 32) * 4
            and_mask = b"\x00" * (mask_row * size)
            # BITMAPINFOHEADER: height is doubled (XOR + AND mask).
            header = struct.pack(
                "<IiiHHIIiiII",
                40,  # biSize
                size,  # biWidth
                size * 2,  # biHeight (xor + and)
                1,  # biPlanes
                32,  # biBitCount
                0,  # biCompression = BI_RGB
                len(bgra),  # biSizeImage
                0,
                0,
                0,
                0,
            )
            entries.append((size, header + bytes(bgra) + and_mask))

    # ICONDIR + ICONDIRENTRY[] + image data.
    n = len(entries)
    out = bytearray()
    out += struct.pack("<HHH", 0, 1, n)  # reserved, type=1 (icon), count
    data_offset = 6 + 16 * n
    image_blob = bytearray()
    for size, payload in entries:
        w = 0 if size >= 256 else size
        h = 0 if size >= 256 else size
        out += struct.pack(
            "<BBBBHHII",
            w,
            h,
            0,  # color count (0 for >= 256 colors / 32bpp)
            0,  # reserved
            1,  # planes
            32,  # bit count
            len(payload),
            data_offset,
        )
        data_offset += len(payload)
        image_blob += payload
    out += image_blob
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(out))
    print(f"Wrote {dst.relative_to(REPO_ROOT)}")


def build_icos() -> None:
    """Generate every multi-size ``.ico`` declared in :data:`ICO_TARGETS`."""
    for src, dst in ICO_TARGETS:
        _build_one_ico(src, dst)


def _svg_to_bmp(src: Path, dst: Path, width: int, height: int) -> None:
    """Rasterise ``src`` (SVG) and save it as a 24 bpp BMP v3 (no alpha).

    WiX UI requires the legacy BMP format (BITMAPINFOHEADER, uncompressed,
    no alpha channel) - which is exactly what Pillow writes by default when
    the input image is in RGB mode.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = _render_svg(src, width, height)
    # Convert RGBA -> RGB (flatten on white) to strip the alpha channel,
    # then save as BMP v3 (24 bpp, uncompressed).
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    bg.save(dst, format="BMP")
    print(f"Wrote {dst.relative_to(REPO_ROOT)}")


def build_wix_bitmaps() -> None:
    """Generate ``wix/dialog.bmp`` and ``wix/banner.bmp`` from their SVG sources."""
    _svg_to_bmp(RESOURCES / "WixUIDialog.svg", WIX / "dialog.bmp", *DIALOG_SIZE)
    _svg_to_bmp(RESOURCES / "WixUIBanner.svg", WIX / "banner.bmp", *BANNER_SIZE)


_NESTED_SVG_HREF_RE = None  # lazy-compiled in :func:`_inline_nested_svgs`


def _inline_nested_svgs(svg_path: Path) -> str:
    """Return the SVG source with ``<image xlink:href="X.svg">`` references
    replaced by base64 PNG data URIs.

    CairoSVG follows ``<image>`` hrefs only for raster formats; it silently
    drops nested SVG references (used by ``DataLab-Frontpage.svg`` and
    ``DataLab-Overview.svg`` to embed ``DataLab-Banner2.svg`` /
    ``DataLab-Panorama.svg``). We rasterise each linked SVG to a PNG sized
    after its declared ``width`` / ``height`` attributes (oversampled 2x for
    quality) and substitute the href in place.
    """
    global _NESTED_SVG_HREF_RE
    if _NESTED_SVG_HREF_RE is None:
        _NESTED_SVG_HREF_RE = re.compile(
            r'(<image\b[^>]*?\b(?:xlink:href|href)\s*=\s*")([^"]+\.svg)(")',
            re.IGNORECASE | re.DOTALL,
        )

    text = svg_path.read_text(encoding="utf-8")

    def _replace(match: "re.Match[str]") -> str:
        href = match.group(2)
        linked = (svg_path.parent / href).resolve()
        if not linked.is_file():
            print(f"  warning: linked SVG not found, leaving as-is: {href}")
            return match.group(0)
        # Render the linked SVG at 2x its declared image size for quality.
        # Pull width/height from the parent <image> element attributes.
        attrs = match.group(0)
        m_w = re.search(r'\bwidth\s*=\s*"([\d.]+)"', attrs)
        # m_h = re.search(r'\bheight\s*=\s*"([\d.]+)"', attrs)
        target_w = int(float(m_w.group(1)) * 2) if m_w else 2000
        target_w = max(800, min(target_w, 6000))
        target_h = _svg_height_for_width(linked, target_w)
        png_bytes = cairosvg.svg2png(
            url=str(linked),
            output_width=target_w,
            output_height=target_h,
        )
        data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode(
            "ascii"
        )
        return f"{match.group(1)}{data_uri}{match.group(3)}"

    return _NESTED_SVG_HREF_RE.sub(_replace, text)


def build_pngs() -> None:
    """Rasterise every PNG declared in :data:`PNG_TARGETS` via CairoSVG.

    CairoSVG (pure Python, libcairo-based) is used here instead of Qt's
    ``QSvgRenderer`` because Qt5 mis-renders the Inkscape-authored logos
    (content scaled by an unexpected ~4/3 factor, cropping the text).
    """
    for src, dst, width in PNG_TARGETS:
        if not src.is_file():
            raise FileNotFoundError(src)
        height = _svg_height_for_width(src, width)
        dst.parent.mkdir(parents=True, exist_ok=True)
        svg_source = _inline_nested_svgs(src)
        cairosvg.svg2png(
            bytestring=svg_source.encode("utf-8"),
            url=str(src),  # base URL for any remaining relative refs
            write_to=str(dst),
            output_width=width,
            output_height=height,
        )
        print(f"Wrote {dst.relative_to(REPO_ROOT)} ({width}x{height})")


def copy_svgs() -> None:
    """Copy plain SVG sources declared in :data:`SVG_COPIES`."""
    for src, dst in SVG_COPIES:
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"Wrote {dst.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    valid = {"ico", "wix", "png", "copy", "all"}
    parser.add_argument(
        "targets",
        nargs="*",
        help=(
            "Subset of resources to regenerate: ico, wix, png, copy, all "
            "(default: all)."
        ),
    )
    args = parser.parse_args(argv)
    targets = set(args.targets) or {"all"}
    unknown = targets - valid
    if unknown:
        parser.error(
            f"invalid target(s): {sorted(unknown)} (choose from {sorted(valid)})"
        )
    if "all" in targets:
        targets = {"ico", "wix", "png", "copy"}

    # A real QApplication is required so QSvgRenderer can access system fonts
    # (Century Gothic, etc.). Do NOT use the "offscreen" platform plugin: it
    # ships without system font integration on Windows.
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    print(f"Using Qt SVG renderer ({type(app).__module__})")

    if "ico" in targets:
        build_icos()
    if "wix" in targets:
        build_wix_bitmaps()
    if "png" in targets:
        build_pngs()
    if "copy" in targets:
        copy_svgs()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
