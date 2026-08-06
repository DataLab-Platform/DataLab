# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Pure unit contracts for mutation-kind history actions."""

from __future__ import annotations

import os
import tempfile

from sigima.objects import SignalROI, create_signal_roi

from datalab.h5.native import NativeH5Writer
from datalab.history.action import HistoryAction
from datalab.history.core import numpy_to_json_safe
from datalab.history.session import HistorySession
from datalab.tests.features.common.history_test_helpers import read_history_sessions


def roi_as_dict(roi: SignalROI) -> dict:
    """Return a plain-Python (comparable) dict representation of a ROI."""
    return numpy_to_json_safe(roi.to_dict())


def build_mutation_action(payload: SignalROI | None) -> HistoryAction:
    """Build a ROI mutation action targeting two signal objects."""
    return HistoryAction(
        title="Edit regions of interest",
        kind=HistoryAction.KIND_MUTATION,
        panel_str="signal",
        mutation_key="roi",
        target_uuids=["first-uuid", "second-uuid"],
        kwargs={"payload": payload},
    )


def roundtrip_action(action: HistoryAction) -> HistoryAction:
    """Serialize an action to HDF5 and return its deserialized counterpart."""
    session = HistorySession(number=1)
    session.add_action(action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        return read_history_sessions(path)[0].actions[0]


def test_mutation_action_hdf5_roundtrip() -> None:
    """Round-trip a ROI mutation action with its payload and identity fields."""
    payload = create_signal_roi([[26, 41], [125, 146]], indices=True)
    action = build_mutation_action(payload)
    loaded = roundtrip_action(action)
    assert loaded.kind == HistoryAction.KIND_MUTATION
    assert loaded.mutation_key == "roi"
    assert loaded.target_uuids == ["first-uuid", "second-uuid"]
    assert loaded.panel_str == "signal"
    loaded_payload = loaded.kwargs.get("payload")
    assert isinstance(loaded_payload, SignalROI)
    assert type(loaded_payload) is type(payload)
    assert roi_as_dict(loaded_payload) == roi_as_dict(payload)


def test_mutation_action_deletion_payload_roundtrip() -> None:
    """Decode a missing payload kwarg as None (ROI deletion)."""
    action = build_mutation_action(None)
    # None kwargs are dropped at construction/encoding time: a deletion
    # mutation simply has no "payload" kwarg on disk.
    assert "payload" not in action.kwargs
    loaded = roundtrip_action(action)
    assert loaded.kind == HistoryAction.KIND_MUTATION
    assert loaded.mutation_key == "roi"
    assert loaded.target_uuids == ["first-uuid", "second-uuid"]
    assert loaded.kwargs.get("payload") is None


def test_mutation_action_copy_and_uuid_remap() -> None:
    """Copy a mutation action independently and remap its target UUIDs."""
    payload = create_signal_roi([[26, 41]], indices=True)
    action = build_mutation_action(payload)
    copied = action.copy()
    assert copied is not action and copied.uuid != action.uuid
    assert copied.mutation_key == "roi"
    assert copied.target_uuids == action.target_uuids
    assert copied.target_uuids is not action.target_uuids
    assert roi_as_dict(copied.kwargs["payload"]) == roi_as_dict(payload)
    remapped = action.copy_with_uuid_remap(
        {"signal": {"first-uuid": "new-first", "second-uuid": "new-second"}}
    )
    assert remapped.target_uuids == ["new-first", "new-second"]
    assert action.target_uuids == ["first-uuid", "second-uuid"]
