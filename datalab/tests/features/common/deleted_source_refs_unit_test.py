# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Deleted source references unit test.

Validates the stable "deleted source reference" registry: when an object or
group referenced in a result title is deleted, its short ID is no longer lost
(it used to degrade to ``sxxx``). Instead, every dependent gets a stable
per-object token (``sd001`` for a deleted signal, ``id001`` for a deleted image,
``gsd001``/``gid001`` for deleted groups) mapped, in the dependent's metadata
registry, to the deleted object's frozen title.

The frozen title is stored in canonical form, so references to *surviving*
sources keep resolving dynamically.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from sigima import create_signal

from datalab.objectmodel import DELETED_REF_KEY, ObjectModel, get_uuid


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


def test_delete_source_freezes_reference() -> None:
    """Deleting a source replaces its short ID by a stable ``sd001`` token."""
    model, gid = _build_model()
    src = _add_signal(model, gid, "My reference")  # s001
    result = _add_signal(model, gid, "fft(s001)")  # s002
    model.remove_object(src)
    # Stored title no longer references the (now invalid) short ID, and no
    # lossy "sxxx" token is produced:
    assert result.title == "fft(sd001)"
    assert "sxxx" not in result.title
    # Short-ID (no-names) display keeps the stable token:
    assert model.get_display_title(result, use_titles=False) == "fft(sd001)"
    # Title display resolves the token to the frozen name:
    assert model.get_display_title(result, use_titles=True) == "fft(My reference)"


def test_registry_is_public_metadata() -> None:
    """The registry lives in a public (copiable) metadata key."""
    model, gid = _build_model()
    src = _add_signal(model, gid, "Ref")  # s001
    result = _add_signal(model, gid, "fft(s001)")  # s002
    model.remove_object(src)
    assert DELETED_REF_KEY in result.metadata
    assert result.metadata[DELETED_REF_KEY] == {"sd001": "Ref"}
    # The registry survives a deep copy of the object (duplication, paste):
    clone = result.copy()
    assert clone.metadata[DELETED_REF_KEY] == {"sd001": "Ref"}


def test_surviving_source_resolves_dynamically() -> None:
    """A frozen title referencing a surviving source keeps resolving live."""
    model, gid = _build_model()
    base = _add_signal(model, gid, "Base")  # s001
    deriv = _add_signal(model, gid, "deriv(s001)")  # s002
    result = _add_signal(model, gid, "fft(s002)")  # s003
    model.remove_object(deriv)
    # The frozen title of the deleted intermediate still references the
    # surviving "Base" object (s001), so the rendered title stays dynamic:
    assert result.title == "fft(sd001)"
    assert model.get_display_title(result, use_titles=True) == "fft(deriv(Base))"
    base.title = "New base"
    assert model.get_display_title(result, use_titles=True) == "fft(deriv(New base))"


def test_multiple_deletions_increment_token() -> None:
    """Deleting several sources of the same object allocates ``sd001``/``sd002``."""
    model, gid = _build_model()
    a = _add_signal(model, gid, "A")  # s001
    b = _add_signal(model, gid, "B")  # s002
    result = _add_signal(model, gid, "s001 + s002")  # s003
    model.remove_object(a)
    model.remove_object(b)
    refs = model.get_deleted_refs(result)
    assert refs == {"sd001": "A", "sd002": "B"}
    assert model.get_display_title(result, use_titles=True) == "A + B"


def test_delete_does_not_affect_other_objects() -> None:
    """Freezing only touches objects that actually reference the deleted one."""
    model, gid = _build_model()
    src = _add_signal(model, gid, "Ref")  # s001
    unrelated = _add_signal(model, gid, "standalone")  # s002
    _add_signal(model, gid, "fft(s001)")  # s003
    model.remove_object(src)
    assert unrelated.title == "standalone"
    assert DELETED_REF_KEY not in unrelated.metadata


def test_delete_group_freezes_group_and_objects() -> None:
    """Deleting a group freezes both the group token and its objects' tokens."""
    model, _gid = _build_model()
    g1 = model.get_group(_gid)  # gs001
    g2 = model.add_group("G2")  # gs002
    g2id = get_uuid(g2)
    inside = _add_signal(model, _gid, "Inside")  # s001 (in g1)
    uses_group = _add_signal(model, g2id, "merge(gs001)")  # references group g1
    uses_obj = _add_signal(model, g2id, "fft(s001)")  # references inside object
    assert inside is not None
    g1_title = g1.title
    model.remove_group(g1)
    # Group reference frozen as a "gsd001" token:
    assert uses_group.title == "merge(gsd001)"
    assert model.get_deleted_refs(uses_group) == {"gsd001": g1_title}
    assert model.get_display_title(uses_group, use_titles=True) == f"merge({g1_title})"
    # Inner object reference frozen as an "sd001" token:
    assert uses_obj.title == "fft(sd001)"
    assert model.get_display_title(uses_obj, use_titles=True) == "fft(Inside)"


def test_no_sxxx_after_deletion() -> None:
    """The legacy lossy ``sxxx`` fallback is never produced on deletion."""
    model, gid = _build_model()
    src = _add_signal(model, gid, "Ref")  # s001
    r1 = _add_signal(model, gid, "fft(s001)")  # s002
    r2 = _add_signal(model, gid, "psd(s001)")  # s003
    model.remove_object(src)
    for result in (r1, r2):
        assert "sxxx" not in result.title
        assert "ixxx" not in result.title


if __name__ == "__main__":
    test_delete_source_freezes_reference()
    test_registry_is_public_metadata()
    test_surviving_source_resolves_dynamically()
    test_multiple_deletions_increment_token()
    test_delete_does_not_affect_other_objects()
    test_delete_group_freezes_group_and_objects()
    test_no_sxxx_after_deletion()
    print("All deleted-source-reference tests passed.")
