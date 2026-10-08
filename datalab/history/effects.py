# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Analysis effects manifest: capture metadata/ROI mutations of 1-to-0 analyses."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np
from sigima.objects.base import ROI_KEY

from datalab.history.core import encode_roi

# Private bookkeeping keys excluded from the metadata diff (ROI changes are
# tracked separately through ``roi_modified``, not the metadata diff)
EXCLUDED_METADATA_KEYS = frozenset({"__uuid", "__number", ROI_KEY})


@dataclass
class AnalysisEffects:
    """Manifest of the side effects produced by a 1-to-0 analysis on one object.

    Attributes:
        metadata_added: Metadata keys created by the analysis.
        metadata_replaced: Pre-existing metadata keys whose value changed.
        roi_modified: True when the analysis created or changed the object's ROI.
        roi_before: Encoded ROI (:func:`datalab.history.core.encode_roi`) of
         the object before the **first** execution; ``""`` when the object had
         no ROI, ``None`` when not recorded (legacy manifests).
    """

    metadata_added: list[str] = field(default_factory=list)
    metadata_replaced: list[str] = field(default_factory=list)
    roi_modified: bool = False
    roi_before: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation.

        The ``roi_before`` key is only present when recorded, so legacy
        payloads round-trip unchanged.

        Returns:
            Dictionary suitable for ``json.dumps`` round-trip.
        """
        data: dict[str, Any] = {
            "metadata_added": list(self.metadata_added),
            "metadata_replaced": list(self.metadata_replaced),
            "roi_modified": bool(self.roi_modified),
        }
        if self.roi_before is not None:
            data["roi_before"] = self.roi_before
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisEffects:
        """Build an instance from a dictionary produced by :meth:`to_dict`.

        Args:
            data: Dictionary payload (missing fields fall back to defaults).

        Returns:
            New :class:`AnalysisEffects` instance.
        """
        return cls(
            metadata_added=list(data.get("metadata_added", [])),
            metadata_replaced=list(data.get("metadata_replaced", [])),
            roi_modified=bool(data.get("roi_modified", False)),
            roi_before=data.get("roi_before"),
        )


def safe_equal(value1: Any, value2: Any) -> bool:
    """Return True if both values compare equal, tolerating numpy arrays.

    Numpy arrays are compared with :func:`numpy.array_equal`. Any comparison
    failure is treated as "changed" (returns False).

    Args:
        value1: First value.
        value2: Second value.

    Returns:
        True if values are considered equal.
    """
    if isinstance(value1, np.ndarray) or isinstance(value2, np.ndarray):
        try:
            return bool(np.array_equal(value1, value2))
        except (TypeError, ValueError):
            return False
    try:
        return bool(value1 == value2)
    except Exception:  # pylint: disable=broad-except
        return False


def merge_effects(
    previous: AnalysisEffects | None, new: AnalysisEffects
) -> AnalysisEffects:
    """Merge a freshly captured manifest into the previous one.

    Keys produced on the first run stay in ``metadata_added`` even though a
    recompute observes them as replaced (the analysis owns them for their
    whole lifetime). ``roi_modified`` is sticky: once an execution touched
    the ROI, the merged manifest keeps the flag. ``roi_before`` is first-run
    sticky: the ROI recorded before the first execution is never overwritten
    by recompute captures. Output lists are sorted for deterministic ordering.

    Args:
        previous: Manifest from earlier executions, or None on first merge.
        new: Manifest captured during the latest execution.

    Returns:
        Merged :class:`AnalysisEffects` instance.
    """
    if previous is None:
        return AnalysisEffects(
            metadata_added=sorted(new.metadata_added),
            metadata_replaced=sorted(new.metadata_replaced),
            roi_modified=new.roi_modified,
            roi_before=new.roi_before,
        )
    added = set(previous.metadata_added) | set(new.metadata_added)
    replaced = (set(previous.metadata_replaced) | set(new.metadata_replaced)) - added
    return AnalysisEffects(
        metadata_added=sorted(added),
        metadata_replaced=sorted(replaced),
        roi_modified=previous.roi_modified or new.roi_modified,
        roi_before=(
            previous.roi_before if previous.roi_before is not None else new.roi_before
        ),
    )


@contextmanager
def capture_effects(obj: Any) -> Generator[AnalysisEffects, None, None]:
    """Capture metadata and ROI mutations applied to an object.

    Snapshots the object's metadata keys/values and ROI before yielding, then
    fills the yielded :class:`AnalysisEffects` instance on exit. Private
    bookkeeping keys (``__uuid``, ``__number``) and the ROI metadata key are
    excluded from the diff. When the ROI was modified, the pre-execution ROI
    is kept in ``roi_before`` (encoded payload) so a later replay can restore
    it before re-running the analysis.

    Args:
        obj: Signal or image object whose ``metadata`` and ``roi`` are watched.

    Yields:
        Mutable :class:`AnalysisEffects` instance, filled on exit.
    """
    before = {
        key: value
        for key, value in obj.metadata.items()
        if key not in EXCLUDED_METADATA_KEYS
    }
    roi_before = obj.roi.copy() if obj.roi is not None else None
    effects = AnalysisEffects()
    try:
        yield effects
    finally:
        after = {
            key: value
            for key, value in obj.metadata.items()
            if key not in EXCLUDED_METADATA_KEYS
        }
        effects.metadata_added = sorted(set(after) - set(before))
        effects.metadata_replaced = sorted(
            key
            for key in set(before) & set(after)
            if not safe_equal(before[key], after[key])
        )
        effects.roi_modified = not safe_equal(roi_before, obj.roi)
        if effects.roi_modified:
            try:
                effects.roi_before = (
                    "" if roi_before is None else encode_roi(roi_before)
                )
            except (TypeError, ValueError):
                pass  # Unencodable ROI: degrade to the flag-only legacy behavior
