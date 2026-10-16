# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Workspace provenance
====================

The :class:`ProvenanceService` records, in a workspace-level ledger, the signal and
image processing executed by DataLab (1-to-1, 1-to-n, 2-to-1, n-to-1 and
analyses), whatever the History "Record" setting. It is the provenance recorder
of DataLab-Capsule (:class:`datalab_capsule.recorder.Recorder`), shared with
DataLab-Web, given the accesses of DataLab Desktop: the ledger model,
fingerprints, replay preparation, comparison and reports come from
DataLab-Capsule; operation contracts and the scientific execution path come from
Sigima. Only qualified operations can be replayed.

Capture never breaks processing: if recording fails, the result is kept, no
activity is recorded and the failure is logged and counted.

.. autoclass:: ProvenanceService
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator, Iterable
from typing import Any

import sigima.proc.contracts as sigima_contracts
from datalab_capsule.ledger import Ledger
from datalab_capsule.recorder import (
    PendingActivity,
    Recorder,
    RecorderHost,
    computation_from_sigima,
)
from sigima.objects import ImageObj, SignalObj

import datalab
from datalab.gui.processor.geometry_postprocess import unwrap_computation_function
from datalab.objectmodel import get_uuid

__all__ = ["EDITION", "PendingActivity", "ProvenanceService"]

EDITION = "desktop"


def object_kind(obj: Any) -> str | None:
    """Return ``"signal"``, ``"image"`` or None for anything else."""
    if isinstance(obj, SignalObj):
        return "signal"
    if isinstance(obj, ImageObj):
        return "image"
    return None


class ProvenanceService(Recorder):
    """Workspace provenance ledger of DataLab Desktop.

    Args:
        find_object: Object lookup by UUID across panels (injected, so that this
         module never depends on panels or History).
        iter_objects: Signals and images of the workspace, in panel order.
        is_known_feature: ``(feature name, plugin origin) -> bool``: whether a
         processing feature named in old metadata is registered.
    """

    def __init__(
        self,
        find_object: Callable[[str], Any],
        iter_objects: Callable[[], Iterable[SignalObj | ImageObj]],
        is_known_feature: Callable[[str, Any], bool],
    ) -> None:
        host = RecorderHost(
            edition=EDITION,
            edition_version=lambda: datalab.__version__,
            object_uuid=get_uuid,
            find_object=find_object,
            iter_objects=iter_objects,
            object_kind=object_kind,
            is_known_operation=lambda _kind, name, origin: is_known_feature(
                name, origin
            ),
        )
        super().__init__(host, computation_from_sigima(sigima_contracts))
        #: Called (without argument) after activities were recorded, e.g. to
        #: refresh the views showing the provenance of objects.
        self.on_recorded: Callable[[], None] | None = None
        self._collectors: list[list[str]] = []

    @contextlib.contextmanager
    def collecting(self) -> Generator[list[str], None, None]:
        """Collect the identifiers of the activities recorded in this scope.

        The identifiers are those returned by the capture itself (completed
        executions, analyses, executions without result, and deferred
        activities once finalized), in recording order: nothing is matched by
        name, date or output UUID. Scopes may be nested.

        Yields:
            The list receiving the identifiers.
        """
        ids: list[str] = []
        self._collectors.append(ids)
        try:
            yield ids
        finally:
            for index, collector in enumerate(self._collectors):
                if collector is ids:
                    del self._collectors[index]
                    break

    def _notify_recorded(self, activity_ids: list[str]) -> None:
        """Collect recorded activities and call :attr:`on_recorded`."""
        if not activity_ids:
            return
        for collector in self._collectors:
            collector.extend(activity_ids)
        if self.on_recorded is not None:
            self.on_recorded()

    def complete(self, pending: Any, output: Any) -> dict[str, Any] | None:
        """Record a completed execution (see ``Recorder.complete``)."""
        activity = super().complete(pending, output)
        self._notify_recorded([activity["activity_id"]] if activity else [])
        return activity

    def complete_analysis(
        self, pending: Any, obj: Any, kind: str, key: str
    ) -> dict[str, Any] | None:
        """Record an analysis (see ``Recorder.complete_analysis``)."""
        activity = super().complete_analysis(pending, obj, kind, key)
        self._notify_recorded([activity["activity_id"]] if activity else [])
        return activity

    def complete_without_result(self, pending: Any) -> dict[str, Any] | None:
        """Record an execution without result (see ``Recorder``)."""
        activity = super().complete_without_result(pending)
        self._notify_recorded([activity["activity_id"]] if activity else [])
        return activity

    def finalize_deferred(self, deferred: Any, remap: dict[str, str]) -> int:
        """Record held activities on their final UUIDs (see ``Recorder``)."""
        start = len(deferred.activity_ids)
        recorded = super().finalize_deferred(deferred, remap)
        self._notify_recorded(list(deferred.activity_ids[start:]))
        return recorded

    def begin(
        self, func: Callable, param: Any, source: Any, *args, **kwargs
    ) -> PendingActivity | None:
        """Record the input states of an execution, before it runs.

        The Sigima function run by a wrapped feature function is recorded (see
        :func:`~datalab.gui.processor.geometry_postprocess.
        unwrap_computation_function`). See
        :meth:`datalab_capsule.recorder.Recorder.begin` for the arguments.
        """
        return super().begin(
            unwrap_computation_function(func), param, source, *args, **kwargs
        )

    def load(
        self,
        block: tuple[Ledger, dict[str, str]] | None,
        replaced: bool,
        saved_version: str | None = None,
        current_version: str | None = None,
        appended: Iterable[Any] = (),
    ) -> None:
        """Adopt the ledger read from a workspace file, after its objects.

        Processing records of objects unknown to the ledger (workspaces saved
        without provenance) are then imported as ``legacy_metadata`` activities.

        Args:
            block: ``(ledger, state_status)`` read before the workspace was
             replaced, or None when the file has no provenance block.
            replaced: False when the file was appended to the current workspace:
             ledgers are not merged and the file's block is ignored.
            saved_version: DataLab version that saved the file, as stored.
            current_version: Version compared with *saved_version* (default:
             this DataLab version).
            appended: Objects added by an appended file: those holding an
             identifier already known to the ledger never inherit its records.
        """
        super().load(
            block,
            replaced,
            saved_version,
            datalab.__version__ if current_version is None else current_version,
            appended=appended,
        )
