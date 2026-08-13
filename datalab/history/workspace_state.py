# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Workspace state snapshot captured at history action time."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from datalab.history.core import numpy_to_json_safe
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.h5.native import NativeH5Reader, NativeH5Writer


class WorkspaceState:
    """Object representing the workspace state at a given time.

    The workspace state stores the per-panel selection of objects by **UUID**
    (robust against reordering, renaming or interleaved insertions). Selections
    remain separate from structured metadata captured for every object in each
    data panel. Informative shape and title lists cover the selected objects.
    """

    def __init__(self) -> None:
        """Create a new workspace state"""
        # The selection is stored as a dictionary where the key is the panel name
        # and the value is the list of UUIDs of selected objects.
        self.selection: dict[str, list[str]] = {}
        # The states are stored as a dictionary where the key is the panel name
        # and the value is the list of states (str) of the objects in the panel. The
        # state is a string containing the object data shape (kept for informative
        # display only -- not used for selection matching anymore).
        self.states: dict[str, list[str]] = {}
        # The titles are stored as a dictionary where the key is the panel name and the
        # value is the list of titles of the objects in the panel. The title is only
        # informative and is not used to determine if two objects have the same state.
        self.titles: dict[str, list[str]] = {}
        # Structured data signatures of all objects, keyed by panel name and UUID.
        # Compatibility checks use entries for selected objects. Missing metadata
        # means a pre-Gate-2 history and falls back to UUID-existence validation.
        self.object_metadata: dict[str, dict[str, dict[str, Any]]] = {}

    def copy(self) -> WorkspaceState:
        """Return an independent copy of this workspace state."""
        state = WorkspaceState()
        state.selection = deepcopy(self.selection)
        state.states = deepcopy(self.states)
        state.titles = deepcopy(self.titles)
        state.object_metadata = deepcopy(self.object_metadata)
        return state

    def serialize(self, writer: NativeH5Writer) -> None:
        """Serialize this workspace state

        Args:
            writer: Writer
        """
        with writer.group("selection"):
            writer.write_dict(self.selection)
        with writer.group("states"):
            writer.write_dict(self.states)
        with writer.group("titles"):
            writer.write_dict(self.titles)
        with writer.group("object_metadata"):
            writer.write_dict(self.object_metadata)

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this workspace state

        Args:
            reader: Reader
        """
        with reader.group("selection"):
            self.selection = reader.read_dict()
        with reader.group("states"):
            self.states = reader.read_dict()
        with reader.group("titles"):
            self.titles = reader.read_dict()
        current = reader.h5
        for option in reader.option:
            current = current[option]
        if "object_metadata" in current.attrs or "object_metadata" in current:
            with reader.group("object_metadata"):
                self.object_metadata = reader.read_dict()
        else:
            self.object_metadata = {}
        # Normalize legacy translated keys to stable panel identifiers.
        self.selection = self.normalize_panel_keys(self.selection)
        self.states = self.normalize_panel_keys(self.states)
        self.titles = self.normalize_panel_keys(self.titles)
        self.object_metadata = self.normalize_panel_keys(self.object_metadata)

    def get_current_selection(self, mainwindow: DLMainWindow) -> dict[str, list[str]]:
        """Get the current selection in the workspace, keyed by panel name and
        valued by the list of selected object UUIDs.

        Args:
            mainwindow: DataLab's main window

        Returns:
            Current selection in the workspace, by panel name → list of UUIDs.
        """
        selection: dict[str, list[str]] = {}
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            selection[panel.PANEL_STR_ID] = [
                get_uuid(obj)
                for obj in panel.objview.get_sel_objects(include_groups=True)
            ]
        return selection

    @staticmethod
    def get_roi_signature(obj: Any) -> str | None:
        """Return a short stable hash of the object's ROI, or None.

        Args:
            obj: Signal or image object (any object exposing a ``roi``
             attribute with a ``to_dict()`` method).

        Returns:
            16-character hex digest of the JSON-encoded ROI, or ``None``
            when the object has no ROI or its encoding fails.
        """
        roi = getattr(obj, "roi", None)
        if roi is None:
            return None
        try:
            payload = numpy_to_json_safe(roi.to_dict())
            serialized = json.dumps(payload, sort_keys=True)
        except (AttributeError, TypeError, ValueError):
            # Defensive: ROI classes evolve; omit signature on failure.
            return None
        return hashlib.sha1(serialized.encode()).hexdigest()[:16]

    @staticmethod
    def get_object_metadata(obj: Any) -> dict[str, Any]:
        """Return a stable data signature for an object."""
        data = getattr(obj, "data", None)
        shape = getattr(data, "shape", None)
        if shape is None:
            return {}
        shape = [int(size) for size in shape]
        ndim = getattr(data, "ndim", len(shape))
        metadata: dict[str, Any] = {"shape": shape, "ndim": int(ndim)}
        roi_signature = WorkspaceState.get_roi_signature(obj)
        if roi_signature is not None:
            metadata["roi"] = roi_signature
        return metadata

    @staticmethod
    def normalize_object_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Normalize object metadata loaded from HDF5 for comparison."""
        shape = metadata.get("shape")
        if shape is None:
            return {}
        shape = [int(size) for size in shape]
        ndim = metadata.get("ndim", len(shape))
        normalized: dict[str, Any] = {"shape": shape, "ndim": int(ndim)}
        roi = metadata.get("roi")
        if roi is not None:
            normalized["roi"] = str(roi)
        return normalized

    # Mapping from legacy translated panel keys to stable identifiers.
    # Covers the English translations; other locales are handled by the
    # catch-all ``"signal"``/``"image"`` substring heuristic below.
    _LEGACY_PANEL_KEY_MAP: dict[str, str] = {
        "Signal Panel": "signal",
        "Image Panel": "image",
    }

    @classmethod
    def normalize_panel_key(cls, key: str) -> str:
        """Map a potentially translated panel key to its stable identifier."""
        if key in ("signal", "image"):
            return key
        mapped = cls._LEGACY_PANEL_KEY_MAP.get(key)
        if mapped is not None:
            return mapped
        # Heuristic for non-English translations: look for the stable ID
        # substring in the key (e.g. "Panneau signal" → "signal").
        lowered = key.lower()
        for stable_id in ("signal", "image"):
            if stable_id in lowered:
                return stable_id
        return key

    @classmethod
    def normalize_panel_keys(cls, d: dict) -> dict:
        """Return *d* with all top-level keys normalized to stable panel IDs."""
        return {cls.normalize_panel_key(k): v for k, v in d.items()}

    def save(self, mainwindow: DLMainWindow, panel_str: str | None = None) -> None:
        """Save the current workspace state

        Args:
            mainwindow: DataLab's main window
            panel_str: Stable identifier (``"signal"`` or ``"image"``) of the
             panel the action operates on. When provided, only that panel's
             selection/states/titles are captured -- the other panel's entries
             are left empty so that unrelated objects in the other panel cannot
             produce false incompatibilities. When ``None`` (default), both
             panels are captured (backward-compatible behavior). ``object_metadata``
             is always captured for both panels (used by session-replay
             positional fallback and harmless for compatibility checks).
        """
        full_selection = self.get_current_selection(mainwindow)
        if panel_str is None:
            self.selection = full_selection
        else:
            # Restrict the captured selection to the action's own panel; leave
            # the other panel's selection empty so compatibility checks ignore
            # unrelated objects in that panel.
            self.selection = {panel_str: full_selection.get(panel_str, [])}
        self.object_metadata = {}
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            sel_uuids = self.selection.get(panel.PANEL_STR_ID, [])
            self.states[panel.PANEL_STR_ID] = [
                str(getattr(getattr(obj, "data", None), "shape", ""))
                for obj in panel.objmodel
                if get_uuid(obj) in sel_uuids
            ]
            self.titles[panel.PANEL_STR_ID] = [
                obj.title for obj in panel.objmodel if get_uuid(obj) in sel_uuids
            ]
            # Store metadata for ALL panel objects (not just selected) so that
            # the dict key order captures the full panel ordering.  During
            # session replay the key order lets us sort old UUIDs by their
            # original panel position, which prevents non-commutative 2_to_1
            # operand swaps in the positional-fallback code path.
            # ``is_current_state_compatible`` only checks *selected* UUIDs, so
            # the extra entries are harmless for compatibility validation.
            self.object_metadata[panel.PANEL_STR_ID] = {
                get_uuid(obj): self.get_object_metadata(obj) for obj in panel.objmodel
            }

    def is_current_state_compatible(
        self,
        mainwindow: DLMainWindow,
        ignore_roi_uuids: set[str] | None = None,
    ) -> bool:
        """Check if the current workspace state is compatible with the saved state.

        Compatibility means that **every** UUID recorded in the saved selection
        still exists in the corresponding panel. When structured object metadata
        is available (current schema), each selected object's data shape,
        dimensions and ROI signature (when recorded) must also match the saved
        signature. Histories without this metadata fall back to legacy
        UUID-existence validation; metadata recorded before ROI signatures
        existed skips the ROI comparison.

        Args:
            mainwindow: DataLab's main window
            ignore_roi_uuids: UUIDs whose ROI signature must be excluded from
             the comparison (both saved and current sides). Used by mutation
             actions whose state was captured after the mutation was applied:
             the target's ROI legitimately differs at replay time.

        Returns:
            True if every saved UUID still exists in its panel and saved
            metadata, when available, still matches.
        """
        if not self.selection:
            return True
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            saved_uuids = self.selection.get(panel.PANEL_STR_ID, [])
            existing_uuids = set(panel.objmodel.get_object_ids())
            saved_metadata = self.object_metadata.get(panel.PANEL_STR_ID, {})
            for uuid in saved_uuids:
                if uuid not in existing_uuids:
                    return False
                if uuid in saved_metadata:
                    current = self.get_object_metadata(panel.objmodel[uuid])
                    current = self.normalize_object_metadata(current)
                    saved = self.normalize_object_metadata(saved_metadata[uuid])
                    if ignore_roi_uuids and uuid in ignore_roi_uuids:
                        # Mutation target: the ROI is precisely what the
                        # action changes, exclude it from both sides.
                        current.pop("roi", None)
                        saved.pop("roi", None)
                    elif "roi" not in saved:
                        # Legacy tolerance: states recorded before the ROI
                        # signature existed must not flag incompatibility.
                        current.pop("roi", None)
                    if saved and current != saved:
                        return False
        return True

    def restore(
        self, mainwindow: DLMainWindow, ignore_roi_uuids: set[str] | None = None
    ) -> None:
        """Restore the workspace state by selecting the recorded UUIDs.

        Args:
            mainwindow: DataLab's main window
            ignore_roi_uuids: UUIDs whose ROI signature is excluded from the
             compatibility check (see :meth:`is_current_state_compatible`).

        Raises:
            ValueError: If a saved UUID no longer exists in its panel or its
             saved metadata (shape/dimensions) is incompatible.
        """
        if not self.selection:
            return
        if not self.is_current_state_compatible(
            mainwindow, ignore_roi_uuids=ignore_roi_uuids
        ):
            raise ValueError(
                "Current workspace state is not compatible with saved state"
            )
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            uuids = self.selection.get(panel.PANEL_STR_ID, [])
            if uuids:
                panel.objview.select_objects(uuids)
