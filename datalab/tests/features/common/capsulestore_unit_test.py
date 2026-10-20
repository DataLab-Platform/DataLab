# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Capsule revisions kept on this computer and registry settings."""

from __future__ import annotations

import json
import os.path as osp

import pytest

from datalab import capsulestore
from datalab.capsulestore import (
    CapsuleRevision,
    RevisionStore,
    get_registry_settings,
    registry_base,
    set_registry_settings,
)
from datalab.config import Conf


def _revision(digit: str, source: str = "exported", **kwargs) -> CapsuleRevision:
    return CapsuleRevision(
        fingerprint="sha256:" + digit * 64,
        name=f"rev-{digit}",
        revision_of=kwargs.get("revision_of"),
        date_published="2026-10-03T08:00:00.000Z",
        source=source,
        manifest={"name": f"rev-{digit}"},
    )


def test_record_and_list(tmp_path) -> None:
    """Revisions are listed most recent first; the first source is kept."""
    store = RevisionStore(str(tmp_path / "revisions"))
    assert store.revisions() == []
    store.record(_revision("1"))
    store.record(_revision("2", "inspected", revision_of="sha256:" + "1" * 64))
    assert [r.name for r in store.revisions()] == ["rev-2", "rev-1"]
    store.record(_revision("1", "registry"))
    first = store.revisions()[0]
    assert (first.name, first.source) == ("rev-1", "exported")
    assert store.get("sha256:" + "2" * 64).revision_of == "sha256:" + "1" * 64
    assert store.get("sha256:" + "3" * 64) is None
    store.delete("sha256:" + "1" * 64)
    store.delete("sha256:" + "1" * 64)
    assert [r.name for r in store.revisions()] == ["rev-2"]


def test_unreadable_and_invalid(tmp_path) -> None:
    """Unreadable files are ignored; fingerprints never become paths."""
    store = RevisionStore(str(tmp_path))
    store.record(_revision("a"))
    with open(osp.join(tmp_path, "b" * 64 + ".json"), "w", encoding="utf-8") as f:
        f.write("{not json")
    with open(osp.join(tmp_path, "c" * 64 + ".json"), "w", encoding="utf-8") as f:
        json.dump({"unexpected": True}, f)
    assert [r.name for r in store.revisions()] == ["rev-a"]
    with pytest.raises(ValueError):
        store.get("sha256:../../etc")


def test_oldest_revisions_are_dropped(tmp_path, monkeypatch) -> None:
    """At most MAX_REVISIONS revisions are kept."""
    monkeypatch.setattr(capsulestore, "MAX_REVISIONS", 2)
    store = RevisionStore(str(tmp_path))
    for digit in "123":
        store.record(_revision(digit))
    assert [r.name for r in store.revisions()] == ["rev-3", "rev-2"]


def test_registry_settings() -> None:
    """The address is saved normalised; the token stays in memory."""
    previous = get_registry_settings()
    try:
        set_registry_settings(" https://registry.example.org/ ", "secret")
        assert get_registry_settings() == ("https://registry.example.org", "secret")
        assert Conf.capsule_registry_address.get() == "https://registry.example.org"
        set_registry_settings("", "")
        assert get_registry_settings() == ("", "")
        with pytest.raises(ValueError):
            set_registry_settings("ftp://registry.example.org", "")
        with pytest.raises(ValueError):
            registry_base("registry.example.org")
    finally:
        set_registry_settings(*previous)
