# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the macro-support utilities:
- :mod:`datalab.utils.recentmacros`
- :mod:`datalab.utils.macrorecovery`
- :mod:`datalab.gui.macros_templates`
"""

from __future__ import annotations

import os
import os.path as osp

import pytest

from datalab.gui.macros_templates import get_template, list_templates
from datalab.utils import macrorecovery, recentmacros

# ---------------------------------------------------------------------------
# Common fixture: redirect cache files to a temporary directory
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Redirect every cache file used by the macro utilities to ``tmp_path``.

    This keeps tests isolated from the user's real DataLab configuration.
    """

    def fake_path(filename: str) -> str:
        return str(tmp_path / filename)

    # Patch the Configuration.get_path() lookup used by both modules.
    monkeypatch.setattr(
        "datalab.utils.recentmacros.Configuration.get_path",
        staticmethod(fake_path),
    )
    monkeypatch.setattr(
        "datalab.utils.macrorecovery.Configuration.get_path",
        staticmethod(fake_path),
    )
    yield


# ---------------------------------------------------------------------------
# recentmacros
# ---------------------------------------------------------------------------


def test_recentmacros_record_and_list():
    """A recorded macro can be retrieved and is listed first."""
    entry = recentmacros.record_recent("My macro", "print('hi')", source="test")
    assert entry["uid"]
    assert entry["title"] == "My macro"
    assert entry["source"] == "test"

    listed = recentmacros.list_recent()
    assert len(listed) == 1
    assert listed[0]["uid"] == entry["uid"]


def test_recentmacros_dedup_by_code_hash():
    """Re-recording the same code refreshes the entry instead of duplicating."""
    first = recentmacros.record_recent("Title A", "code body", source="a")
    second = recentmacros.record_recent("Title B", "code body", source="b")
    # Same uid (deduped), but title/source/last_seen are refreshed.
    assert first["uid"] == second["uid"]
    assert second["title"] == "Title B"
    assert second["source"] == "b"
    assert len(recentmacros.list_recent()) == 1


def test_recentmacros_get_and_remove():
    """get_recent finds by uid; remove_recent deletes it."""
    entry = recentmacros.record_recent("X", "x = 1")
    assert recentmacros.get_recent(entry["uid"]) is not None
    assert recentmacros.remove_recent(entry["uid"]) is True
    assert recentmacros.get_recent(entry["uid"]) is None
    assert recentmacros.remove_recent("nonexistent") is False


def test_recentmacros_clear():
    """clear_recent removes all entries."""
    recentmacros.record_recent("A", "a = 1")
    recentmacros.record_recent("B", "b = 2")
    assert len(recentmacros.list_recent()) == 2
    recentmacros.clear_recent()
    assert recentmacros.list_recent() == []


def test_recentmacros_max_entries_cap():
    """The cache is trimmed to MAX_ENTRIES (oldest dropped)."""
    for i in range(recentmacros.MAX_ENTRIES + 5):
        recentmacros.record_recent(f"M{i}", f"x = {i}")
    listed = recentmacros.list_recent(limit=10_000)
    assert len(listed) == recentmacros.MAX_ENTRIES


def test_recentmacros_corrupted_file_returns_empty():
    """A corrupted JSON file is silently ignored."""
    path = recentmacros._path()  # pylint: disable=protected-access
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fdesc:
        fdesc.write("not valid json {{{")
    assert recentmacros.list_recent() == []


# ---------------------------------------------------------------------------
# macrorecovery
# ---------------------------------------------------------------------------


def test_macrorecovery_save_and_load():
    """A saved pending entry is retrievable by uid."""
    macrorecovery.save_pending("uid-1", "Draft", "draft code")
    pending = macrorecovery.load_pending()
    assert "uid-1" in pending
    assert pending["uid-1"]["title"] == "Draft"
    assert pending["uid-1"]["code"] == "draft code"


def test_macrorecovery_overwrite_same_uid():
    """Saving twice with the same uid overwrites the previous entry."""
    macrorecovery.save_pending("uid-1", "Old", "old code")
    macrorecovery.save_pending("uid-1", "New", "new code")
    pending = macrorecovery.load_pending()
    assert len(pending) == 1
    assert pending["uid-1"]["code"] == "new code"


def test_macrorecovery_clear_single():
    """clear_pending(uid) removes only that entry."""
    macrorecovery.save_pending("a", "A", "code a")
    macrorecovery.save_pending("b", "B", "code b")
    macrorecovery.clear_pending("a")
    pending = macrorecovery.load_pending()
    assert "a" not in pending
    assert "b" in pending


def test_macrorecovery_clear_all():
    """clear_pending() removes every entry."""
    macrorecovery.save_pending("a", "A", "code a")
    macrorecovery.save_pending("b", "B", "code b")
    macrorecovery.clear_pending()
    assert macrorecovery.load_pending() == {}


def test_macrorecovery_atomic_write_no_tmp_leftover(tmp_path):
    """The on-disk file is the atomic target, not the ``.tmp`` staging file."""
    macrorecovery.save_pending("uid-1", "T", "code")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "recent_macros_pending.json" in files
    assert not any(name.endswith(".tmp") for name in files)


# ---------------------------------------------------------------------------
# macros_templates
# ---------------------------------------------------------------------------


def test_templates_list_has_expected_entries():
    """The bundled template package exposes the documented templates."""
    templates = list_templates()
    names = {t.name for t in templates}
    assert {"simple_macro", "imageproc_macro", "call_method_macro"}.issubset(names)


def test_templates_have_code_and_description():
    """Each template has a non-empty code body and a parsed description."""
    for template in list_templates():
        assert template.title
        assert template.code.strip(), f"Empty code in template {template.name!r}"
        # description may be empty if the marker is absent, but for our bundled
        # templates it should be present.
        assert template.description, (
            f"Missing description in template {template.name!r}"
        )


def test_templates_get_by_name():
    """get_template returns the matching MacroTemplate (or None)."""
    template = get_template("simple_macro")
    assert template is not None
    assert template.name == "simple_macro"
    assert get_template("does_not_exist") is None
