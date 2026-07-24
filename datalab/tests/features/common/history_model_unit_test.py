# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Pure unit contracts for history persistence, copying and replay mapping."""

from __future__ import annotations

import os
import tempfile

from datalab.h5.native import NativeH5Writer
from datalab.history.action import HistoryAction
from datalab.history.core import HISTORY_ACTION_SCHEMA_VERSION, HISTORY_SCHEMA_VERSION
from datalab.history.session import HistorySession
from datalab.tests.features.common.history_test_helpers import (
    build_history_action,
    build_replay_map,
    build_workspace_state,
    delete_hdf5_items_by_name,
    read_history_sessions,
)


def test_action_hdf5_current_and_legacy_contract() -> None:
    """Round-trip current fields and apply all legacy defaults."""
    action = build_history_action()
    action.plugin_origin = {
        "module": "example.plugin",
        "metadata": {"entry_points": ["difference"]},
    }
    action.snapshot_kwargs()
    action.kwargs["pairwise"] = True
    session = HistorySession(number=1)
    session.add_action(action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        current = read_history_sessions(path)[0].actions[0]
        for attribute in ("selection", "states", "titles"):
            values = getattr(action.state, attribute)
            setattr(action.state, attribute, {"Signal Panel": values["signal"]})
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
            for field in ("schema_version", "uuid", "saved_kwargs", "output_uuids"):
                delete_hdf5_items_by_name(writer.h5, field)
            delete_hdf5_items_by_name(writer.h5, "object_metadata")
        legacy = read_history_sessions(path)[0].actions[0]
    assert current.uuid == action.uuid
    assert current.schema_version == HISTORY_ACTION_SCHEMA_VERSION
    assert current.output_uuids == ["output-uuid"]
    assert current.plugin_origin == action.plugin_origin
    assert current.has_pending_edits and bool(current.kwargs["pairwise"])
    assert legacy.schema_version == HISTORY_SCHEMA_VERSION
    assert legacy.uuid != action.uuid and legacy.output_uuids == []
    assert not legacy.has_pending_edits and legacy.state.object_metadata == {}
    assert legacy.state.selection == {"signal": ["source-uuid"]}
    assert legacy.state.states == {"signal": ["(10,)"]}
    assert legacy.state.titles == {"signal": ["Source"]}


def test_action_copy_remaps_all_uuid_references() -> None:
    """Copy an action independently and rewrite every captured UUID field."""
    action = build_history_action()
    action.plugin_origin = {
        "module": "example.plugin",
        "metadata": {"entry_points": ["difference"]},
    }
    copied = action.copy_with_uuid_remap(
        {
            "signal": {
                "source-uuid": "new-source",
                "second-uuid": "new-second",
                "output-uuid": "new-output",
            }
        }
    )
    assert copied is not action and copied.uuid != action.uuid
    assert copied.state.selection == {"signal": ["new-source"]}
    assert copied.state.object_metadata == {
        "signal": {"new-source": {"shape": [10], "ndim": 1, "title": "Source"}}
    }
    assert copied.kwargs["obj2_uuids"] == "new-second"
    assert copied.output_uuids == ["new-output"]
    assert copied.plugin_origin == action.plugin_origin
    copied.state.object_metadata["signal"]["new-source"]["shape"] = [20]
    copied.plugin_origin["metadata"]["entry_points"].append("average")
    assert action.state.object_metadata["signal"]["source-uuid"]["shape"] == [10]
    assert action.plugin_origin["metadata"]["entry_points"] == ["difference"]


def test_replay_uuid_map_matches_exact_title_and_position() -> None:
    """Match replay inputs by exact UUID, unique title, then panel position."""
    replay_map, _signal_model, _image_model = build_replay_map(
        [("same", "Exact"), ("new-position", "Other"), ("new-title", "Named")]
    )
    replay_map.unclaimed["signal"] = ["same", "new-position", "new-title"]
    state = build_workspace_state(
        ["same", "old-title", "old-position"], ["Exact", "Named", "Recorded"]
    )
    action = HistoryAction(
        kind=HistoryAction.KIND_COMPUTE, panel_str="signal", state=state
    )
    replay_map.claim_action_inputs(action)
    assert replay_map.mapping["signal"] == {
        "same": "same",
        "old-title": "new-title",
        "old-position": "new-position",
    }
    assert replay_map.unclaimed["signal"] == []


def test_replay_uuid_map_preserves_operands_and_tracks_changes() -> None:
    """Claim obj2 before primary inputs and track panel-local object changes."""
    replay_map, signal_model, image_model = build_replay_map(
        [("new-second", "Second"), ("new-primary", "Primary")],
        [("image-old", "Image")],
    )
    replay_map.unclaimed["signal"] = ["new-second", "new-primary"]
    action = HistoryAction(
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="signal",
        pattern="2_to_1",
        kwargs={"obj2_uuids": ["old-second"]},
        state=build_workspace_state(["old-primary"], ["Primary"]),
    )
    replay_map.claim_action_inputs(action)
    assert replay_map.mapping["signal"] == {
        "old-second": "new-second",
        "old-primary": "new-primary",
    }
    before = replay_map.snapshot_object_ids()
    signal_model.add("signal-new", "Created")
    replay_map.capture_changes(
        HistoryAction(
            kind=HistoryAction.KIND_UI,
            state=build_workspace_state(["old-new"]),
        ),
        before,
    )
    assert replay_map.mapping["signal"]["old-new"] == "signal-new"
    before = replay_map.snapshot_object_ids()
    signal_model.remove("signal-new")
    replay_map.capture_changes(HistoryAction(), before)
    assert "old-new" not in replay_map.mapping["signal"]
    assert image_model.get_object_ids() == ["image-old"]
