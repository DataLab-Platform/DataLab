# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Title reference unit test.

Validates the helpers underpinning UUID8 title references:

- :func:`datalab.objectmodel.find_title_references`
- :func:`datalab.objectmodel.remap_title_references`
- :func:`datalab.widgets.titledelegate._build_html`
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import sigima.proc.signal as sips
from sigima.objects import create_signal
from sigima.tests.data import create_paracetamol_signal

from datalab.config import Conf
from datalab.objectmodel import (
    ObjectModel,
    find_title_references,
    get_short_uuid,
    get_title_reference,
    patch_title_with_ids,
    remap_title_references,
)
from datalab.tests import datalab_test_app_context
from datalab.widgets.titledelegate import REFERENCE_URL_SCHEME, _build_html


def test_title_references_use_uuid8() -> None:
    """Objects are referenced by UUID8, groups by ``g`` + UUID8."""
    model = ObjectModel("gs")
    group = model.add_group("Group")
    src1, src2 = create_signal("A"), create_signal("B")
    assert get_title_reference(src1) == get_short_uuid(src1)
    assert get_title_reference(group) == f"g{get_short_uuid(group)}"
    dst = create_signal("{0}-{1}")
    patch_title_with_ids(dst, [src1, src2])
    assert dst.title == f"{get_short_uuid(src1)}-{get_short_uuid(src2)}"


def test_find_title_references() -> None:
    """Object and group references are found; other words are ignored."""
    matches = find_title_references("avg(1a2b3c4d, g5e6f7a8b)|s001 x1a2b3c4d 123456789")
    assert [match[2] for match in matches] == ["1a2b3c4d", "g5e6f7a8b"]


def test_remap_title_references_only_replaces_mapped_references() -> None:
    """Unmapped tokens, legacy short IDs and user text are left unchanged."""
    title = "fft(1a2b3c4d) of g5e6f7a8b, run 20260107, s001"
    remapped = remap_title_references(
        title, {"1a2b3c4d": "9f8e7d6c", "g5e6f7a8b": "g0a1b2c3d"}
    )
    assert remapped == "fft(9f8e7d6c) of g0a1b2c3d, run 20260107, s001"


def test_build_html_links_only_resolved_references() -> None:
    """Resolved references become links, unresolved ones stay plain text."""
    html = _build_html("1a2b3c4d-20260107", {"1a2b3c4d": "Signal A"}.get)
    assert f'<a href="{REFERENCE_URL_SCHEME}:1a2b3c4d">Signal A</a>' in html
    assert html.endswith("-20260107")


def test_build_html_escapes_text() -> None:
    """Surrounding text is HTML-escaped to avoid markup injection."""
    html = _build_html("<not a tag> & friends")
    assert html == "&lt;not a tag&gt; &amp; friends"


def test_title_mode_follows_source_renames() -> None:
    """Source-title mode renders the current title of the referenced source."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        source = create_paracetamol_signal()
        panel.add_object(source)
        panel.processor.run_feature(sips.derivative)
        derived = panel.objview.get_current_object()
        assert win.render_object_title(derived.title) == derived.title
        with Conf.result_title_mode.context("title"):
            source.title = "Renamed source"
            assert win.render_object_title(derived.title) == (
                "derivative(Renamed source)"
            )
