# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result title rendering mode unit test.

Validates the display-only rendering of result titles introduced for issue #149:
result/group titles may embed source object short IDs (default, e.g. ``fft(s001)``)
or source object titles (e.g. ``fft(My signal)``), depending on the
``Conf.proc.result_title_mode`` setting.

The stored ``obj.title`` always stays in canonical short-ID form; only the
displayed title changes. Rendering is performed by
:meth:`datalab.objectmodel.ObjectModel.get_display_title`.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import pytest
from sigima import create_signal

from datalab.objectmodel import ObjectModel, get_uuid


def _add_signal(model: ObjectModel, group_id: str, title: str):
    """Create a signal with ``title`` and add it to ``model`` under ``group_id``."""
    sig = create_signal(title, x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 3.0])
    model.add_object(sig, group_id)
    return sig


def _build_model() -> tuple[ObjectModel, str]:
    """Return an empty signal object model and its single group UUID."""
    model = ObjectModel(group_prefix="gs")
    group = model.add_group("Group")
    return model, get_uuid(group)


def test_short_id_mode_returns_stored_title() -> None:
    """In short-ID mode, the display title is the stored (canonical) title."""
    model, gid = _build_model()
    _add_signal(model, gid, "My reference")  # s001
    result = _add_signal(model, gid, "fft(s001)")  # s002
    assert model.get_display_title(result, use_titles=False) == "fft(s001)"
    # The stored title is never altered:
    assert result.title == "fft(s001)"


def test_title_mode_1_to_1() -> None:
    """In title mode, a 1-to-1 result embeds the source object title."""
    model, gid = _build_model()
    _add_signal(model, gid, "My reference")  # s001
    result = _add_signal(model, gid, "fft(s001)")  # s002
    assert model.get_display_title(result, use_titles=True) == "fft(My reference)"
    # Stored title stays canonical:
    assert result.title == "fft(s001)"


def test_title_mode_2_to_1() -> None:
    """In title mode, a 2-to-1 result embeds both source titles."""
    model, gid = _build_model()
    _add_signal(model, gid, "Alpha")  # s001
    _add_signal(model, gid, "Beta")  # s002
    result = _add_signal(model, gid, "s001 + s002")  # s003
    assert model.get_display_title(result, use_titles=True) == "Alpha + Beta"


def test_title_mode_n_to_1() -> None:
    """In title mode, an n-to-1 result embeds every source title."""
    model, gid = _build_model()
    _add_signal(model, gid, "A")  # s001
    _add_signal(model, gid, "B")  # s002
    _add_signal(model, gid, "C")  # s003
    result = _add_signal(model, gid, "average(s001, s002, s003)")  # s004
    assert model.get_display_title(result, use_titles=True) == "average(A, B, C)"


def test_title_mode_group_name() -> None:
    """Group auto-names follow the setting just like object titles."""
    model, gid = _build_model()
    _add_signal(model, gid, "Alpha")  # s001
    _add_signal(model, gid, "Beta")  # s002
    result_group = model.add_group("sum(s001, s002)")
    assert model.get_display_title(result_group, use_titles=True) == "sum(Alpha, Beta)"
    assert model.get_display_title(result_group, use_titles=False) == "sum(s001, s002)"


def test_title_mode_updates_on_rename() -> None:
    """Renaming a source updates the displayed title (stored title unchanged)."""
    model, gid = _build_model()
    src = _add_signal(model, gid, "Old name")  # s001
    result = _add_signal(model, gid, "fft(s001)")  # s002
    assert model.get_display_title(result, use_titles=True) == "fft(Old name)"
    src.title = "New name"
    assert model.get_display_title(result, use_titles=True) == "fft(New name)"
    assert result.title == "fft(s001)"


def test_title_mode_recursive_nesting() -> None:
    """Nested results are rendered recursively down to plain source titles."""
    model, gid = _build_model()
    _add_signal(model, gid, "My base")  # s001
    _add_signal(model, gid, "deriv(s001)")  # s002
    result = _add_signal(model, gid, "fft(s002)")  # s003
    assert model.get_display_title(result, use_titles=True) == "fft(deriv(My base))"


def test_title_mode_missing_source_keeps_token() -> None:
    """An unresolved short ID (removed source) is left untouched."""
    model, gid = _build_model()
    _add_signal(model, gid, "Only one")  # s001
    result = _add_signal(model, gid, "fft(s099)")  # s002 references missing s099
    assert model.get_display_title(result, use_titles=True) == "fft(s099)"


def test_title_mode_cycle_guard() -> None:
    """A self-referential title does not cause infinite recursion.

    This is a pathological case that cannot occur through normal processing
    (results only reference pre-existing objects, forming a DAG). The guard
    guarantees termination: the short ID is left untouched once revisited.
    """
    model, gid = _build_model()
    src = _add_signal(model, gid, "placeholder")  # s001
    # Force a self-reference (cannot happen through normal processing):
    src.title = "loop(s001)"
    # Rendering terminates (no infinite recursion) and keeps the short ID:
    assert model.get_display_title(src, use_titles=True) == "loop(loop(s001))"


def test_lookup_by_stored_title() -> None:
    """An object can be looked up by its stored (canonical) title."""
    model, gid = _build_model()
    _add_signal(model, gid, "My reference")  # s001
    result = _add_signal(model, gid, "fft(s001)")  # s002
    assert model.get_object_from_title("fft(s001)") is result


def test_lookup_by_rendered_title() -> None:
    """An object can be looked up by its rendered (source-title) name.

    This makes macros work when the user references the object by the
    human-readable title shown in the GUI (e.g. ``fft(My reference)``),
    regardless of the current result-title display mode.
    """
    model, gid = _build_model()
    _add_signal(model, gid, "My reference")  # s001
    result = _add_signal(model, gid, "fft(s001)")  # s002
    assert model.get_object_from_title("fft(My reference)") is result


def test_lookup_by_rendered_title_recursive() -> None:
    """Lookup by rendered title works through nested results."""
    model, gid = _build_model()
    _add_signal(model, gid, "My base")  # s001
    _add_signal(model, gid, "deriv(s001)")  # s002
    result = _add_signal(model, gid, "fft(s002)")  # s003
    assert model.get_object_from_title("fft(deriv(My base))") is result


def test_lookup_unknown_title_raises() -> None:
    """Looking up a non-existent title (stored or rendered) raises KeyError."""
    model, gid = _build_model()
    _add_signal(model, gid, "My reference")  # s001
    _add_signal(model, gid, "fft(s001)")  # s002
    with pytest.raises(KeyError):
        model.get_object_from_title("does not exist")


def test_group_lookup_by_rendered_title() -> None:
    """A group can be looked up by its stored and rendered title."""
    model, gid = _build_model()
    src = _add_signal(model, gid, "My reference")  # s001
    group = model.get_group(gid)
    group.title = "group(s001)"
    # Stored-title lookup:
    assert model.get_group_from_title("group(s001)") is group
    # Rendered-title lookup:
    assert model.get_group_from_title("group(My reference)") is group
    assert src is not None  # keep the source alive for short-ID resolution


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
