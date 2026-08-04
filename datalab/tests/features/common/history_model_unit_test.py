# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Pure unit contracts for history persistence, copying and replay mapping."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sigima.objects import Gauss2DParam, ImageObj

from datalab.gui.creation import (
    create_image_from_param,
    extract_creation_parameters,
)
from datalab.gui.panel.history import recompute as hrec
from datalab.gui.processor.base import (
    ProcessingParameters,
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.h5.native import NativeH5Writer
from datalab.history.action import HistoryAction
from datalab.history.core import HISTORY_ACTION_SCHEMA_VERSION, HISTORY_SCHEMA_VERSION
from datalab.history.session import HistorySession
from datalab.history.workspace_state import WorkspaceState
from datalab.objectmodel import get_uuid, set_uuid
from datalab.tests.features.common.history_test_helpers import (
    build_history_action,
    build_replay_map,
    build_workspace_state,
    delete_hdf5_items_by_name,
    read_history_sessions,
)


class CascadeObjectModel:
    """Minimal object model for pure cascade recomputation tests."""

    def __init__(self, objects: list[ImageObj]) -> None:
        self.objects = {get_uuid(obj): obj for obj in objects}

    def __getitem__(self, uuid: str) -> ImageObj:
        """Return the image identified by ``uuid``."""
        return self.objects[uuid]

    def has_uuid(self, uuid: str) -> bool:
        """Return whether ``uuid`` exists in the model."""
        return uuid in self.objects

    def get_object_ids(self) -> list[str]:
        """Return all object UUIDs in insertion order."""
        return list(self.objects)


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


def test_edited_image_creation_recomputes_downstream_in_place() -> None:
    """Regenerate an edited image before recomputing its existing descendant."""
    initial_param = Gauss2DParam.create(
        title="Initial Gaussian",
        height=24,
        width=28,
        x0=-4.0,
        y0=2.0,
        sigma=1.2,
        a=25.0,
    )
    edited_param = Gauss2DParam.create(
        title="Edited Gaussian",
        height=24,
        width=28,
        x0=3.0,
        y0=-2.0,
        sigma=3.5,
        a=80.0,
    )
    source = create_image_from_param(initial_param)
    source_uuid = get_uuid(source)
    source_identity = id(source)
    initial_source_data = source.data.copy()
    expected_source = create_image_from_param(edited_param)

    downstream = source.copy()
    set_uuid(downstream)
    downstream.title = "Initial downstream"
    downstream.data = np.full(source.data.shape, -1.0)
    downstream_uuid = get_uuid(downstream)
    downstream_identity = id(downstream)
    initial_downstream_data = downstream.data.copy()
    insert_processing_parameters(
        downstream,
        ProcessingParameters(
            func_name="unit_transform",
            pattern="1-to-1",
            source_uuid=source_uuid,
        ),
    )

    creation_action = HistoryAction(
        title="Create edited Gaussian",
        kind=HistoryAction.KIND_UI,
        target="imagepanel",
        method_name="new_object",
        kwargs={"param": edited_param},
    )
    creation_action.output_uuids = [source_uuid]
    downstream_state = WorkspaceState()
    downstream_state.selection = {"image": [source_uuid]}
    downstream_action = HistoryAction(
        title="Transform edited Gaussian",
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="image",
        func_name="unit_transform",
        pattern="1_to_1",
        state=downstream_state,
    )
    downstream_action.output_uuids = [downstream_uuid]
    session = HistorySession(number=1)
    session.add_action(creation_action)
    session.add_action(downstream_action)

    processor_source_objects: list[ImageObj] = []
    processor_source_data: list[np.ndarray] = []

    def recompute_1_to_1(
        func_name: str | None,
        source_obj: ImageObj,
        param: object,
        *,
        plugin_origin: dict[str, object] | None,
    ) -> SimpleNamespace:
        assert func_name == "unit_transform"
        assert param is None
        assert plugin_origin is None
        processor_source_objects.append(source_obj)
        processor_source_data.append(source_obj.data.copy())
        new_obj = source_obj.copy()
        new_obj.title = f"unit_transform({source_obj.title})"
        new_obj.data = source_obj.data.astype(float) * 2.0 + 3.0
        return SimpleNamespace(cancelled=False, error_msg=None, result=new_obj)

    def apply_recomputed_object_in_place(
        obj: ImageObj,
        new_obj: ImageObj,
        proc_params: ProcessingParameters,
    ) -> None:
        hrec.update_obj_in_place(obj, new_obj)
        insert_processing_parameters(obj, proc_params)

    object_model = CascadeObjectModel([source, downstream])
    data_panel = SimpleNamespace(
        PANEL_STR_ID="image",
        objmodel=object_model,
        processor=SimpleNamespace(recompute_1_to_1=recompute_1_to_1),
        objprop=SimpleNamespace(
            apply_recomputed_object_in_place=apply_recomputed_object_in_place
        ),
    )
    runtime = SimpleNamespace(
        objects=SimpleNamespace(
            action_output_uuids={
                creation_action.uuid: [source_uuid],
                downstream_action.uuid: [downstream_uuid],
            }
        ),
        execution=SimpleNamespace(cascade_warnings=[], broken_actions=set()),
    )
    history_panel = SimpleNamespace(
        runtime=runtime,
        history_sessions=[session],
    )
    refreshed_uuids: list[str] = []

    def record_refresh(panel_data: object, output_uuid: str) -> None:
        assert panel_data is data_panel
        refreshed_uuids.append(output_uuid)

    descendants = hrec.hchain.get_downstream_actions(history_panel, creation_action)
    assert descendants == [downstream_action]

    with (
        patch.object(hrec.hchain, "resolve_panel_for_action", return_value=data_panel),
        patch.object(
            hrec, "create_image_from_param", wraps=create_image_from_param
        ) as create_image_mock,
        patch.object(hrec, "refresh_target", side_effect=record_refresh),
    ):
        assert hrec.recompute_creation_in_place(history_panel, creation_action)
        assert hrec.recompute_1_to_1_in_place(history_panel, downstream_action)

    create_image_mock.assert_called_once()
    assert create_image_mock.call_args.args[0] is edited_param

    assert id(source) == source_identity
    assert object_model[source_uuid] is source
    assert get_uuid(source) == source_uuid
    np.testing.assert_allclose(source.data, expected_source.data)
    assert not np.array_equal(source.data, initial_source_data)
    creation_param = extract_creation_parameters(source)
    assert isinstance(creation_param, Gauss2DParam)
    for name in ("height", "width", "x0", "y0", "sigma", "a"):
        assert getattr(creation_param, name) == getattr(edited_param, name)

    assert processor_source_objects == [source]
    np.testing.assert_allclose(processor_source_data[0], expected_source.data)
    assert id(downstream) == downstream_identity
    assert object_model[downstream_uuid] is downstream
    assert get_uuid(downstream) == downstream_uuid
    np.testing.assert_allclose(
        downstream.data, expected_source.data.astype(float) * 2.0 + 3.0
    )
    assert not np.array_equal(downstream.data, initial_downstream_data)
    downstream_params = extract_processing_parameters(downstream)
    assert downstream_params is not None
    assert downstream_params.source_uuid == source_uuid
    assert refreshed_uuids == [source_uuid, downstream_uuid]
    assert runtime.execution.cascade_warnings == []
