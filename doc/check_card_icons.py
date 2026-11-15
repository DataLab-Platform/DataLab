# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Compare card-title icons in the English and French HTML documentation.

Run after both Sphinx builds: ``python doc/check_card_icons.py EN_DIR FR_DIR``.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path


class CardIconParser(HTMLParser):
    """Collect the SVG markup of each card title, including icon-free titles."""

    def __init__(self) -> None:
        super().__init__()
        self.titles: list[list[str]] = []
        self._title_depth = 0
        self._svg: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "div":
            if self._title_depth:
                self._title_depth += 1
            elif "sd-card-title" in (dict(attrs).get("class") or "").split():
                self.titles.append([])
                self._title_depth = 1
        if self._title_depth and tag == "svg":
            self._svg = []
        if self._svg is not None:
            self._svg.append(self.get_starttag_text())

    def handle_endtag(self, tag: str) -> None:
        if self._svg is not None:
            self._svg.append(f"</{tag}>")
            if tag == "svg":
                self.titles[-1].append("".join(self._svg))
                self._svg = None
        if tag == "div" and self._title_depth:
            self._title_depth -= 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._svg is not None:
            self._svg.append(self.get_starttag_text())

    def handle_data(self, data: str) -> None:
        if self._svg is not None:
            self._svg.append(data)


def card_icons(path: Path) -> list[list[str]]:
    """Read the card-title SVG markup from one HTML page."""
    parser = CardIconParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser.titles


def main() -> None:
    """Fail when translated card titles lose or change their icons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("english", type=Path)
    parser.add_argument("french", type=Path)
    args = parser.parse_args()
    pages = titles = icons = 0
    for english_path in sorted(args.english.rglob("*.html")):
        english = card_icons(english_path)
        if not english:
            continue
        relative = english_path.relative_to(args.english)
        french = card_icons(args.french / relative)
        if len(english) != len(french):
            raise SystemExit(
                f"{relative}: card title count differs "
                f"(English: {len(english)}, French: {len(french)})"
            )
        for index, (en_icons, fr_icons) in enumerate(zip(english, french), start=1):
            if en_icons != fr_icons:
                raise SystemExit(
                    f"{relative}: card title {index} has missing or changed icons "
                    f"(English: {len(en_icons)}, French: {len(fr_icons)})"
                )
            icons += len(en_icons)
        pages += 1
        titles += len(english)
    if not icons:
        raise SystemExit("No English card-title icons found: check the build paths.")
    print(f"Card icons match: {icons} icons in {titles} titles across {pages} pages.")


if __name__ == "__main__":
    main()
