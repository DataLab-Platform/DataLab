# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Short ID title link unit test.

Validates the helpers underpinning the clickable short-ID feature in
:mod:`datalab.gui.objectview`:

- :func:`datalab.objectmodel.find_short_ids_in_title`
- :func:`datalab.widgets.titledelegate._build_html`
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from datalab.objectmodel import find_short_ids_in_title
from datalab.widgets.titledelegate import (
    SHORT_ID_URL_SCHEME,
    _build_html,
    _links_from_text,
)


def _html(text: str) -> str:
    """Render ``text`` to HTML using the literal short-ID link detection."""
    return _build_html(text, _links_from_text(text))


def test_find_short_ids_in_title_basic() -> None:
    """``find_short_ids_in_title`` extracts every short ID with its bounds."""
    matches = find_short_ids_in_title("s003: average(s001, s002)")
    assert [m[2] for m in matches] == ["s003", "s001", "s002"]
    assert matches[0][0] == 0  # leading short ID starts at offset 0


def test_find_short_ids_in_title_mixed_kinds() -> None:
    """Image, group and signal short IDs are all detected."""
    matches = find_short_ids_in_title("i012: derived(s001, gi003)")
    assert [m[2] for m in matches] == ["i012", "s001", "gi003"]


def test_find_short_ids_in_title_no_false_positives() -> None:
    """Random ``letter+digits`` patterns are not mistaken for short IDs."""
    # `s12345` has too many digits; `s1` has too few.
    assert find_short_ids_in_title("s12345 then s1") == []
    # Substrings inside a word must not match either.
    assert find_short_ids_in_title("class s001abc") == []


def test_build_html_skips_leading_short_id() -> None:
    """The leading ``s001:`` part is rendered as plain text."""
    html = _html("s003: average(s001, s002)")
    # Leading "s003" must NOT be wrapped in an anchor
    assert html.startswith("s003")
    assert f'href="{SHORT_ID_URL_SCHEME}:s003"' not in html
    # But s001 and s002 inside the body must be anchors
    assert f'href="{SHORT_ID_URL_SCHEME}:s001"' in html
    assert f'href="{SHORT_ID_URL_SCHEME}:s002"' in html


def test_build_html_escapes_text() -> None:
    """Surrounding text is HTML-escaped to avoid markup injection."""
    html = _html("s003: <not a tag> & friends")
    assert "&lt;not a tag&gt;" in html
    assert "&amp;" in html


def test_build_html_no_short_ids_returns_plain_text() -> None:
    """Without short IDs, the output is just the escaped text."""
    assert _html("Just a title") == "Just a title"


def test_build_html_uses_explicit_link_spans() -> None:
    """Explicit link spans turn arbitrary substrings (long names) into anchors
    pointing to the given short IDs, even without literal short IDs in the
    text."""
    text = "s003: average(First signal, Second signal)"
    # Spans locate the long names and map them back to the source short IDs:
    s1 = text.index("First signal")
    s2 = text.index("Second signal")
    links = [
        (s1, s1 + len("First signal"), "s001"),
        (s2, s2 + len("Second signal"), "s002"),
    ]
    html = _build_html(text, links)
    assert f'href="{SHORT_ID_URL_SCHEME}:s001">First signal</a>' in html
    assert f'href="{SHORT_ID_URL_SCHEME}:s002">Second signal</a>' in html
    # The leading canonical short ID is left as plain (escaped) text:
    assert html.startswith("s003")
