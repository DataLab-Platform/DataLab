# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Capsule export and opening
==========================

:class:`CapsuleInputOutput` exports the workspace as a capsule (``.dlcapsule``)
and opens a capsule as the workspace. Capsules are created, read and checked by
DataLab-Capsule (:mod:`datalab_capsule.archive`, :mod:`datalab_capsule.exchange`);
this module only saves and opens the workspace through
:class:`~datalab.gui.h5io.H5InputOutput`, keeps the capsule the workspace was
opened from (the parent of a revision exported later) and records the revisions
seen on this computer (:class:`~datalab.capsulestore.RevisionStore`).

.. autoclass:: CapsuleInputOutput
"""

from __future__ import annotations

import mimetypes
import os
import os.path as osp
import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from datalab_capsule import exchange
from datalab_capsule.archive import (
    DEFAULT_SIZE_POLICY,
    MAX_ATTACHMENT_BYTES,
    CapsuleError,
    SizePolicy,
    create_from_hdf5,
    read_capsule,
)
from datalab_capsule.manifest import inspect_manifest

from datalab.capsulestore import CapsuleRevision, RevisionStore
from datalab.config import _

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow

__all__ = [
    "SIZE_POLICY",
    "CapsuleInputOutput",
    "attachment_entries",
    "read_bounded",
    "workspace_leaf",
]

#: Size limits of DataLab Desktop for capsules: the default policy of
#: DataLab-Capsule (4 GiB workspace), an accepted upper bound, not a tested
#: capacity.
SIZE_POLICY = DEFAULT_SIZE_POLICY


def _mib(size: int) -> str:
    return f"{size / 1024**2:.1f}"


def read_bounded(filename: str, limit: int, what: str) -> bytes:
    """Read a whole file, refusing it before reading when it is too large.

    Args:
        filename: File name.
        limit: Largest size accepted (bytes).
        what: What the file is, for the error message (e.g. "Capsule").

    Returns:
        The content of the file.

    Raises:
        CapsuleError: If the file is larger than *limit*, even if it grew while
         being read: no more than *limit* + 1 bytes are read.
    """
    size = os.stat(filename).st_size
    if size <= limit:
        with open(filename, "rb") as file:
            data = file.read(limit + 1)
        if len(data) <= limit:
            return data
        size = len(data)
    raise CapsuleError(
        _("%s too large: %s, %s MiB (limit: %s MiB)")
        % (what, osp.basename(filename), _mib(size), _mib(limit))
    )


def workspace_leaf(workspace_ref: str) -> str:
    """Return the file name of a workspace published at *workspace_ref*."""
    leaf = workspace_ref.rstrip("/").rsplit("/", 1)[-1]
    return leaf if leaf and ":" not in leaf else "workspace.h5"


def attachment_entries(paths: Sequence[str]) -> list[tuple[str, str]]:
    """Return ``(name, path)`` of attachment files, named as in the capsule.

    Names are allowed by the capsule profile and unique, in the order of
    *paths* (see :func:`datalab_capsule.exchange.attachment_name`).
    """
    taken: list[str] = []
    for path in paths:
        taken.append(exchange.attachment_name(osp.basename(path), taken))
    return list(zip(taken, paths))


class CapsuleInputOutput:
    """Capsule export and opening for the DataLab main window.

    Args:
        mainwindow: Main window
        revisions: Revisions seen on this computer (default: the store of the
         DataLab configuration folder).
        policy: Size limits of the capsules read or written (default:
         :data:`SIZE_POLICY`). Files are checked against them before they are
         read, and a capsule this edition would refuse is never written.
    """

    def __init__(
        self,
        mainwindow: DLMainWindow,
        revisions: RevisionStore | None = None,
        policy: SizePolicy | None = None,
    ) -> None:
        self.mainwindow = mainwindow
        self.revisions = RevisionStore() if revisions is None else revisions
        self.policy = SIZE_POLICY if policy is None else policy

    def read_capsule_file(self, filename: str) -> bytes:
        """Read a capsule file, refused before reading beyond the size policy."""
        return read_bounded(filename, self.policy.max_archive_bytes, _("Capsule"))

    def read_workspace_file(self, filename: str) -> bytes:
        """Read a referenced workspace file, refused beyond the size policy."""
        return read_bounded(
            filename, self.policy.max_workspace_bytes, _("Referenced workspace")
        )

    def record_revision(self, manifest: dict[str, Any], source: str) -> None:
        """Record a revision seen on this computer (never fails the caller)."""
        try:
            summary = inspect_manifest(manifest)
            self.revisions.record(
                CapsuleRevision.from_summary(summary, manifest, source)
            )
        except (OSError, ValueError, KeyError):
            pass

    def workspace_bytes(self) -> bytes:
        """Return the workspace saved as a DataLab HDF5 file (with provenance).

        The workspace is saved to a temporary file, as by **Save to HDF5 file**,
        without changing the workspace file name or the last folder used.

        Raises:
            CapsuleError: If the saved workspace exceeds the size policy (checked
             before the file is read back).
        """
        handle, path = tempfile.mkstemp(suffix=".h5", prefix="datalab-capsule-")
        os.close(handle)
        try:
            self.mainwindow.h5inputoutput.save_file(path)
            return read_bounded(path, self.policy.max_workspace_bytes, _("Workspace"))
        finally:
            os.remove(path)

    def export(
        self,
        filename: str,
        name: str | None = None,
        description: str | None = None,
        as_revision: bool = True,
        workspace_ref: str | None = None,
        workspace_filename: str | None = None,
        attachments: Sequence[str] = (),
    ) -> str:
        """Export the workspace as a capsule.

        Args:
            filename: Capsule file name (``.dlcapsule``).
            name: Capsule name.
            description: Capsule description.
            as_revision: When the workspace was opened from a capsule, name it as
             the parent revision.
            workspace_ref: Absolute IRI where the workspace will be published:
             the capsule references it instead of including it, and the
             workspace is saved apart, to *workspace_filename*.
            workspace_filename: File name of the workspace published at
             *workspace_ref* (default: the last segment of the IRI, next to
             the capsule).
            attachments: Light files (protocol, notes, small tables) declared in
             the capsule, never opened or executed.

        Returns:
            Manifest fingerprint of the capsule.

        Raises:
            CapsuleError: If the workspace or an attachment is too large, or the
             capsule cannot be created or would be refused when read back. No
             file is written then.
            ManifestError: If a field is invalid (workspace IRI, attachment).
        """
        files = []
        for attachment, path in attachment_entries(attachments):
            if osp.getsize(path) > MAX_ATTACHMENT_BYTES:
                raise CapsuleError(
                    _("Attachments are limited to 16 MiB: %s") % osp.basename(path)
                )
            with open(path, "rb") as file:
                data = file.read()
            files.append(
                {
                    "name": attachment,
                    "data": data,
                    "media_type": mimetypes.guess_type(path)[0]
                    or "application/octet-stream",
                }
            )
        origin = self.mainwindow.provenance.capsule_origin
        workspace = self.workspace_bytes()
        capsule = create_from_hdf5(
            workspace,
            name=name or None,
            description=description or None,
            revision_of=origin["fingerprint"] if origin and as_revision else None,
            workspace_ref=workspace_ref or None,
            attachments=files,
        )
        # Read back before writing anything: a capsule this edition refuses
        # (size policy, invalid manifest) is never written.
        manifest = read_capsule(capsule, self.policy).manifest
        if workspace_ref:
            if workspace_filename is None:
                workspace_filename = osp.join(
                    osp.dirname(osp.abspath(filename)), workspace_leaf(workspace_ref)
                )
            with open(workspace_filename, "wb") as file:
                file.write(workspace)
        with open(filename, "wb") as file:
            file.write(capsule)
        self.record_revision(manifest, "exported")
        return exchange.capsule_origin(manifest)["fingerprint"]

    def open(self, filename: str, workspace: str | None = None) -> None:
        """Open a capsule as the workspace (replacing it).

        The capsule is checked first (size, archive, manifest, workspace SHA-256
        and size); nothing changes if it is invalid. The capsule and the
        referenced workspace are refused before being read when they exceed the
        size policy.

        Args:
            filename: Capsule file name.
            workspace: Workspace file of a capsule that references it.

        Raises:
            CapsuleError: If the capsule is invalid or too large, or references a
             workspace that was not supplied, is too large or does not match.
            ProvenanceFormatError: If the provenance block is invalid.
        """
        data = self.read_capsule_file(filename)
        supplied = None if workspace is None else self.read_workspace_file(workspace)
        capsule = read_capsule(
            data,
            self.policy,
            resolve_workspace=None if supplied is None else lambda _iri: supplied,
        )
        if capsule.workspace is None:
            raise CapsuleError(
                _("This capsule references its workspace: select the workspace file")
            )
        handle, path = tempfile.mkstemp(suffix=".h5", prefix="datalab-capsule-")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(capsule.workspace)
            self.mainwindow.h5inputoutput.open_file(path, True, True)
        finally:
            os.remove(path)
        self.mainwindow.provenance.capsule_origin = exchange.capsule_origin(
            capsule.manifest
        )
        self.record_revision(capsule.manifest, "inspected")
