# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Bounded renderings of history action parameters (History panel descriptions).

Recorded parameters may hold voluminous values (arrays, long strings, rendering
payloads such as a histogram). The History panel only needs a readable
description of the actual parameters, so every rendering produced here is:

- limited to the persistent parameters: transient items (``data.transient``
  property, e.g. :class:`guidata.dataset.HistogramRangeItem`) carry editor
  context, not dataset state, and are not serialized;
- bounded in length, whatever the size of the underlying values.
"""

from __future__ import annotations

import html

from guidata.dataset.dataitems import ButtonItem, HistogramRangeItem
from guidata.dataset.datatypes import (
    BeginGroup,
    DataItem,
    DataSet,
    EndGroup,
    ObjectItem,
    SeparatorItem,
)

ELLIPSIS = "…"
#: Maximum length of each value in the collapsed (single-line) summary
SUMMARY_VALUE_MAX_LENGTH = 80
#: Maximum length of the whole collapsed summary
SUMMARY_MAX_LENGTH = 300
#: Maximum length of each value in the expanded (HTML) description
DETAILS_VALUE_MAX_LENGTH = 1000
#: Maximum length of a plain text rendered in the expanded description
DETAILS_TEXT_MAX_LENGTH = 5000

# Items that structure the dataset form but hold no parameter value
STRUCTURAL_ITEM_TYPES = (BeginGroup, EndGroup, SeparatorItem, ButtonItem)

# Same style as guidata's ``DataSet.to_html`` (readable in light and dark modes)
TITLE_STYLE = "color: #5294e2"
LABEL_CELL_STYLE = "text-align: right; vertical-align: top;"
VALUE_CELL_STYLE = "text-align: left; padding-left: 10px;"
NAME_CELL_STYLE = (
    "text-align: left; padding-left: 15px; "
    "color: #888; font-family: monospace; font-size: 0.9em;"
)


def shorten(text: str, max_length: int) -> str:
    """Return ``text`` truncated to ``max_length`` characters, ellipsis included.

    Args:
        text: Text to shorten
        max_length: Maximum length of the returned text

    Returns:
        The text itself if short enough, otherwise its beginning followed by an
         ellipsis
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(ELLIPSIS)].rstrip() + ELLIPSIS


def single_line(text: str) -> str:
    """Return ``text`` on a single line, with whitespace runs collapsed.

    Args:
        text: Text to flatten

    Returns:
        Single-line text
    """
    return " ".join(text.split())


def first_line(text: str) -> str:
    """Return the first non-empty line of ``text`` (stripped), or an empty string.

    Args:
        text: Multi-line text

    Returns:
        First non-empty line
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def text_to_html(text: str, max_length: int = DETAILS_TEXT_MAX_LENGTH) -> str:
    """Return plain ``text`` as escaped HTML, shortened before escaping.

    Truncating before escaping never cuts an HTML entity or tag.

    Args:
        text: Plain text
        max_length: Maximum length of the text (before escaping)

    Returns:
        HTML text, line breaks rendered as ``<br>``
    """
    return "<br>".join(html.escape(shorten(text, max_length)).splitlines())


def is_transient(item: DataItem) -> bool:
    """Return True if ``item`` holds transient (non-persistent) context.

    Args:
        item: Dataset item

    Returns:
        True for items whose ``data.transient`` property is set
    """
    return bool(item.get_prop("data", "transient", False))


def summarize_dataset(param: DataSet) -> str:
    """Return a compact single-line summary of the parameters of ``param``.

    The summary is the dataset title followed by ``name=value`` pairs for its
    public, persistent items. Each value is shortened to
    :data:`SUMMARY_VALUE_MAX_LENGTH` characters.

    Args:
        param: Parameters

    Returns:
        Summary, or an empty string if the dataset has neither title nor value
    """
    title = param.get_title() or ""
    pairs: list[str] = []
    for item in param.get_items():
        name = item.get_name()
        if (
            not name
            or name.startswith("_")
            or isinstance(item, STRUCTURAL_ITEM_TYPES)
            or is_transient(item)
        ):
            continue
        try:
            value = item.get_value(param)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        value_str = f"{value:g}" if isinstance(value, float) else str(value)
        value_str = shorten(single_line(value_str), SUMMARY_VALUE_MAX_LENGTH)
        pairs.append(f"{name}={value_str}")
    if pairs:
        return f"{title}: {', '.join(pairs)}" if title else ", ".join(pairs)
    return title


def _item_row(param: DataSet, item: DataItem, value_html: str) -> str:
    """Return the HTML table row describing ``item``."""
    label = item.get_prop_value("display", param, "label")
    if not label:
        # BoolItem without label: use its text as label
        label = item.get_prop_value("display", param, "text", "")
    name = html.escape(item.get_name())
    return (
        f'<tr><td style="{LABEL_CELL_STYLE}">{label}:</td>'
        f'<td style="{VALUE_CELL_STYLE}">{value_html}</td>'
        f'<td style="{NAME_CELL_STYLE}">({name})</td></tr>'
    )


def _value_row(param: DataSet, item: DataItem) -> str:
    """Return the HTML table row describing the value of ``item``."""
    value_str = item.get_string_value(param)
    return _item_row(param, item, text_to_html(value_str, DETAILS_VALUE_MAX_LENGTH))


def dataset_to_html(param: DataSet) -> str:
    """Return a bounded HTML description of the parameters of ``param``.

    The rendering mirrors :meth:`guidata.dataset.DataSet.to_html` (title and
    comment, then a table of labels, values and names), except that:

    - transient items are omitted, and the range edited through a
      :class:`guidata.dataset.HistogramRangeItem` is shown with its linked
      fields, even though they are hidden from the parameter form;
    - each value is HTML-escaped and shortened to
      :data:`DETAILS_VALUE_MAX_LENGTH` characters.

    Args:
        param: Parameters

    Returns:
        HTML description
    """
    text = ""
    title = param.get_title()
    if title:
        text = f'<u><b style="{TITLE_STYLE}">{title}</b></u>:'
    comment = param.get_comment()
    if comment:
        text += f'<br><span style="{TITLE_STYLE}">{comment}</span>'
    rows: list[str] = []
    shown: set[str] = set()
    for item in param.get_items():
        if isinstance(item, STRUCTURAL_ITEM_TYPES):
            continue
        if isinstance(item, HistogramRangeItem):
            try:
                linked_items = item.get_range_items(param)
            except ValueError:
                continue
            for linked_item in linked_items:
                if linked_item.get_name() not in shown:
                    shown.add(linked_item.get_name())
                    rows.append(_value_row(param, linked_item))
            continue
        if (
            is_transient(item)
            or item.get_name() in shown
            or bool(item.get_prop_value("display", param, "hide", False))
        ):
            continue
        shown.add(item.get_name())
        if isinstance(item, ObjectItem):
            value = item.get_value(param)
            value_html = dataset_to_html(value) if isinstance(value, DataSet) else "-"
            rows.append(_item_row(param, item, value_html))
        else:
            rows.append(_value_row(param, item))
    if rows:
        text += '<table border="0">' + "".join(rows) + "</table>"
    return text
