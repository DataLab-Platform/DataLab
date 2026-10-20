# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Capsule revisions and registry settings
=======================================

Local state of the capsule dialog, kept apart from the GUI:

- :class:`RevisionStore`: flat list of the capsule revisions exported, inspected
  or retrieved from a registry on this computer, one JSON file per revision in
  the DataLab configuration folder. Each revision keeps its manifest, so that
  any two revisions can be compared later without the capsule files. It is a
  local convenience, never a source of truth: the capsule files are.
- Registry settings: the address is saved in the configuration, the publishing
  token is only kept in memory for the session.

.. autoclass:: CapsuleRevision
.. autoclass:: RevisionStore
"""

from __future__ import annotations

import dataclasses
import json
import os
import os.path as osp
import time
import urllib.parse
from typing import Any, Literal

from datalab.config import Conf, _, get_config_path

__all__ = [
    "MAX_REVISIONS",
    "CapsuleRevision",
    "RevisionStore",
    "get_registry_settings",
    "registry_base",
    "set_registry_settings",
]

#: Largest number of revisions kept; the oldest recorded are dropped first.
MAX_REVISIONS = 100

RevisionSource = Literal["exported", "inspected", "registry"]


@dataclasses.dataclass
class CapsuleRevision:
    """A capsule revision seen on this computer.

    Attributes:
        fingerprint: Manifest fingerprint (``sha256:…``).
        name: Capsule name.
        revision_of: Fingerprint of the parent revision, or None.
        date_published: Publication date of the manifest.
        source: How the revision reached this computer.
        manifest: Capsule manifest.
        recorded_at: When it was recorded here (seconds since epoch).
    """

    fingerprint: str
    name: str
    revision_of: str | None
    date_published: str | None
    source: RevisionSource
    manifest: dict[str, Any]
    recorded_at: float = 0.0

    @classmethod
    def from_summary(
        cls, summary: dict[str, Any], manifest: dict[str, Any], source: RevisionSource
    ) -> CapsuleRevision:
        """Return the revision of a manifest summary (``inspect_manifest``)."""
        return cls(
            fingerprint=summary["fingerprint"],
            name=summary["name"] or "",
            revision_of=summary["revision_of"],
            date_published=summary["date_published"],
            source=source,
            manifest=manifest,
        )


class RevisionStore:
    """Capsule revisions kept on this computer, one JSON file per revision.

    Args:
        directory: Folder of the revision files (default: ``capsule_revisions``
         in the DataLab configuration folder).
    """

    def __init__(self, directory: str | None = None) -> None:
        self.directory = (
            get_config_path("capsule_revisions") if directory is None else directory
        )

    def _path(self, fingerprint: str) -> str:
        digest = fingerprint.split(":")[-1]
        if not digest or not all(c in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid fingerprint: {fingerprint!r}")
        return osp.join(self.directory, f"{digest}.json")

    def _read(self, path: str) -> CapsuleRevision | None:
        try:
            with open(path, encoding="utf-8") as file:
                return CapsuleRevision(**json.load(file))
        except (OSError, ValueError, TypeError):
            return None

    def record(self, revision: CapsuleRevision) -> None:
        """Record a revision; one already recorded keeps its first ``source``."""
        path = self._path(revision.fingerprint)
        existing = self._read(path) if osp.isfile(path) else None
        recorded = self.revisions()
        # Strictly after the others, even with a coarse system clock.
        newest = max((r.recorded_at for r in recorded), default=0.0)
        revision = dataclasses.replace(
            revision,
            source=revision.source if existing is None else existing.source,
            recorded_at=max(time.time(), newest + 1e-6),
        )
        os.makedirs(self.directory, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(dataclasses.asdict(revision), file, ensure_ascii=False)
        os.replace(temporary, path)
        for old in self.revisions()[MAX_REVISIONS:]:
            self.delete(old.fingerprint)

    def revisions(self) -> list[CapsuleRevision]:
        """Return the revisions, most recently recorded first.

        Unreadable files are ignored.
        """
        if not osp.isdir(self.directory):
            return []
        revisions = [
            revision
            for name in os.listdir(self.directory)
            if name.endswith(".json")
            and (revision := self._read(osp.join(self.directory, name))) is not None
        ]
        return sorted(revisions, key=lambda r: r.recorded_at, reverse=True)

    def get(self, fingerprint: str) -> CapsuleRevision | None:
        """Return one revision, or None."""
        path = self._path(fingerprint)
        return self._read(path) if osp.isfile(path) else None

    def delete(self, fingerprint: str) -> None:
        """Forget one revision."""
        try:
            os.remove(self._path(fingerprint))
        except FileNotFoundError:
            pass


# Publishing token: memory only, forgotten when DataLab closes.
_TOKEN = ""


def registry_base(address: str) -> str:
    """Normalise a registry address; only absolute http(s) URLs are accepted.

    Raises:
        ValueError: If *address* is not an http or https URL.
    """
    parsed = urllib.parse.urlparse(address.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(_("The registry address must be an http or https URL"))
    return urllib.parse.urlunparse(parsed).rstrip("/")


def get_registry_settings() -> tuple[str, str]:
    """Return ``(address, token)``; "" when no registry or no token is set."""
    return Conf.capsule_registry_address.get(), _TOKEN


def set_registry_settings(address: str, token: str) -> None:
    """Save the registry address and keep the token for this session.

    Raises:
        ValueError: If *address* is not empty and not an http(s) URL.
    """
    global _TOKEN  # pylint: disable=global-statement
    address = address.strip()
    Conf.capsule_registry_address.set(registry_base(address) if address else "")
    _TOKEN = token
