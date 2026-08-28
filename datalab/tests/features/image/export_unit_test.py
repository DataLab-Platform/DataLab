# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Format-aware image export unit tests."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest
from sigima.io import ImageExportParam

from datalab.gui.panel import base as panel_base
from datalab.gui.panel import image as image_panel
from datalab.gui.panel.base import BaseDataPanel
from datalab.gui.panel.image import (
    ImageExportGUIParam,
    ImagePanel,
    create_image_export_gui_param,
)
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.history.action import HistoryAction
from datalab.history.core import decode_kwargs, encode_kwargs
from datalab.history.session import HistorySession


class SavePanelStub:
    """Minimal panel implementing the collaborators used by ``save_to_files``."""

    PANEL_STR_ID = "image"

    def __init__(
        self, objects: list[object], selected_objects: list[object] | None = None
    ) -> None:
        selected_objects = objects if selected_objects is None else selected_objects
        self.objview = SimpleNamespace(
            get_sel_objects=Mock(return_value=selected_objects)
        )
        objects_by_uuid = {obj.metadata["__uuid"]: obj for obj in objects}
        self.objmodel = MagicMock()
        self.objmodel.has_uuid.side_effect = objects_by_uuid.__contains__
        self.objmodel.__getitem__.side_effect = objects_by_uuid.__getitem__
        io_registry = SimpleNamespace(
            get_write_filters=Mock(return_value="Images (*.png)")
        )
        setattr(self, "IO_REGISTRY", io_registry)
        setattr(self, "parentWidget", self.parent_widget)
        self.edit_export_parameters = Mock(return_value=(True, None))
        self.write_object_to_file = Mock()
        self.historypanel = SimpleNamespace(add_ui_entry=Mock())
        self.mainwindow = SimpleNamespace(historypanel=self.historypanel)

    def parent_widget(self):
        """Return the parent expected by the generic save flow."""
        return None


def create_stub_object(uuid: str) -> SimpleNamespace:
    """Return a minimal object with a stable DataLab UUID."""
    return SimpleNamespace(metadata={"__uuid": uuid})


@contextmanager
def suppress_write_error(*_args):
    """Suppress an expected writer error like ``qt_try_loadsave_file``."""
    try:
        yield
    except OSError:
        pass


def get_dtype_choice_keys(param) -> list[str]:
    """Return the target dtype keys exposed by an export parameter."""
    item = next(item for item in param.get_items() if item.get_name() == "target_dtype")
    choices = item.get_prop_value("data", param, "choices")
    return [key for key, _label, _icon in choices]


@pytest.mark.parametrize(
    ("extension", "expected_choices", "expected_normalizations"),
    [
        ("png", ["auto", "uint8"], ["minmax", "minmax", "minmax"]),
        ("jpeg", ["auto", "uint8"], ["minmax", "minmax", "minmax"]),
        ("bmp", ["auto", "uint8"], ["minmax", "minmax", "minmax"]),
        (
            "jp2",
            ["auto", "uint8", "uint16"],
            ["none", "minmax", "minmax"],
        ),
        (
            "tiff",
            ["auto", "uint8", "uint16", "float32", "float64"],
            ["none", "none", "none"],
        ),
        (
            "npy",
            ["auto", "uint8", "uint16", "float32", "float64"],
            ["none", "none", "none"],
        ),
    ],
)
def test_image_export_choices_and_defaults_are_format_aware(
    extension: str,
    expected_choices: list[str],
    expected_normalizations: list[str],
) -> None:
    """Check format defaults for representative unsigned, signed and float data."""
    for source_dtype, expected_normalization in zip(
        (np.uint16, np.int32, np.float32), expected_normalizations
    ):
        source_data = np.array([[1, 3]], dtype=source_dtype)
        image = SimpleNamespace(data=source_data)

        param = create_image_export_gui_param(image, f"image.{extension}")

        assert get_dtype_choice_keys(param) == expected_choices
        assert param.normalization == expected_normalization
        assert param.behavior == "rescale"
        assert param.target_dtype == "auto"
        np.testing.assert_array_equal(image.data, source_data)


def test_interactive_export_cancellation_skips_write_and_history_pair() -> None:
    """Exclude a cancelled image export dialog from writing and history."""
    objects = [create_stub_object("uuid-1"), create_stub_object("uuid-2")]
    panel = SavePanelStub(objects)
    confirmed_param = ImageExportParam.create(normalization="minmax")
    panel.edit_export_parameters.side_effect = [
        (False, None),
        (True, confirmed_param),
    ]

    with (
        patch.object(
            panel_base,
            "getsavefilename",
            side_effect=[("cancelled.png", ""), ("confirmed.png", "")],
        ),
        patch.object(panel_base, "save_restore_stds", side_effect=nullcontext),
        patch.object(
            panel_base,
            "qt_try_loadsave_file",
            side_effect=lambda *_args: nullcontext(),
        ),
        patch.object(panel_base.Conf.base_dir, "get", return_value=""),
        patch.object(panel_base.Conf.base_dir, "set"),
    ):
        BaseDataPanel.save_to_files(panel)

    panel.write_object_to_file.assert_called_once_with(
        objects[1], "confirmed.png", confirmed_param
    )
    history_call = panel.historypanel.add_ui_entry.call_args
    assert history_call.kwargs["filenames"] == ["confirmed.png"]
    assert history_call.kwargs["export_params"] == [confirmed_param]
    assert history_call.kwargs["object_uuids"] == ["uuid-2"]


def test_object_uuids_resolve_exports_independently_of_selection() -> None:
    """Resolve replay exports by recorded UUID and preserve their order."""
    first = create_stub_object("uuid-1")
    second = create_stub_object("uuid-2")
    unrelated = create_stub_object("uuid-unrelated")
    panel = SavePanelStub([first, second, unrelated], selected_objects=[unrelated])

    with (
        patch.object(
            panel_base,
            "qt_try_loadsave_file",
            side_effect=lambda *_args: nullcontext(),
        ),
        patch.object(panel_base.Conf.base_dir, "set"),
    ):
        BaseDataPanel.save_to_files(
            panel,
            ["second.png", "first.png"],
            object_uuids=["uuid-2", "uuid-1"],
        )

    panel.objview.get_sel_objects.assert_not_called()
    panel.write_object_to_file.assert_has_calls(
        [call(second, "second.png", None), call(first, "first.png", None)]
    )
    history_call = panel.historypanel.add_ui_entry.call_args
    assert history_call.kwargs["filenames"] == ["second.png", "first.png"]
    assert history_call.kwargs["object_uuids"] == ["uuid-2", "uuid-1"]


def test_missing_object_uuid_is_warned_and_skipped_without_realigning() -> None:
    """Skip a stale UUID while preserving each surviving export tuple."""
    first = create_stub_object("uuid-1")
    second = create_stub_object("uuid-2")
    panel = SavePanelStub([first, second])
    first_param = ImageExportParam.create(normalization="none")
    second_param = ImageExportParam.create(normalization="minmax")
    stale_param = ImageExportParam.create(normalization="percentile")

    with (
        patch.object(
            panel_base,
            "qt_try_loadsave_file",
            side_effect=lambda *_args: nullcontext(),
        ),
        patch.object(panel_base.Conf.base_dir, "set"),
        pytest.warns(UserWarning, match="uuid-missing"),
    ):
        BaseDataPanel.save_to_files(
            panel,
            ["second.png", "missing.png", "first.png"],
            [second_param, stale_param, first_param],
            ["uuid-2", "uuid-missing", "uuid-1"],
        )

    panel.write_object_to_file.assert_has_calls(
        [
            call(second, "second.png", second_param),
            call(first, "first.png", first_param),
        ]
    )
    history_call = panel.historypanel.add_ui_entry.call_args
    assert history_call.kwargs["filenames"] == ["second.png", "first.png"]
    assert history_call.kwargs["export_params"] == [second_param, first_param]
    assert history_call.kwargs["object_uuids"] == ["uuid-2", "uuid-1"]


def test_all_missing_object_uuids_write_and_record_nothing() -> None:
    """Do not write or record an export when all recorded UUIDs are stale."""
    panel = SavePanelStub([])

    with pytest.warns(UserWarning, match="no longer exists") as caught:
        BaseDataPanel.save_to_files(
            panel,
            ["first.png", "second.png"],
            object_uuids=["uuid-missing-1", "uuid-missing-2"],
        )

    assert len(caught) == 2
    panel.write_object_to_file.assert_not_called()
    panel.historypanel.add_ui_entry.assert_not_called()


def test_failed_write_is_excluded_from_history() -> None:
    """Record only the tuple whose writer returned successfully."""
    objects = [create_stub_object("uuid-1"), create_stub_object("uuid-2")]
    panel = SavePanelStub(objects)
    panel.write_object_to_file.side_effect = [OSError("failed"), None]

    with (
        patch.object(
            panel_base,
            "qt_try_loadsave_file",
            side_effect=suppress_write_error,
        ),
        patch.object(panel_base.Conf.base_dir, "set"),
    ):
        BaseDataPanel.save_to_files(panel, ["failed.png", "saved.png"])

    history_call = panel.historypanel.add_ui_entry.call_args
    assert history_call.kwargs["filenames"] == ["saved.png"]
    assert history_call.kwargs["object_uuids"] == ["uuid-2"]
    assert "export_params" not in history_call.kwargs


def test_all_failed_writes_create_no_history_action() -> None:
    """Do not record an action when every writer call fails."""
    objects = [create_stub_object("uuid-1"), create_stub_object("uuid-2")]
    panel = SavePanelStub(objects)
    panel.write_object_to_file.side_effect = OSError("failed")

    with (
        patch.object(
            panel_base,
            "qt_try_loadsave_file",
            side_effect=suppress_write_error,
        ),
        patch.object(panel_base.Conf.base_dir, "set"),
    ):
        BaseDataPanel.save_to_files(panel, ["failed-1.png", "failed-2.png"])

    panel.historypanel.add_ui_entry.assert_not_called()


def test_explicit_filename_is_noninteractive_and_uses_legacy_path() -> None:
    """Keep explicit saves noninteractive when no export parameter is supplied."""
    obj = create_stub_object("uuid-1")
    panel = SavePanelStub([obj])

    with (
        patch.object(
            panel_base,
            "qt_try_loadsave_file",
            side_effect=lambda *_args: nullcontext(),
        ),
        patch.object(panel_base.Conf.base_dir, "set"),
    ):
        BaseDataPanel.save_to_files(panel, ["explicit.png"])

    panel.edit_export_parameters.assert_not_called()
    panel.write_object_to_file.assert_called_once_with(obj, "explicit.png", None)
    history_call = panel.historypanel.add_ui_entry.call_args
    assert history_call.kwargs["filenames"] == ["explicit.png"]
    assert history_call.kwargs["object_uuids"] == ["uuid-1"]
    assert "export_params" not in history_call.kwargs


@pytest.mark.parametrize("extension", ["h5", "csv", "txt", "asc", "mat"])
def test_unsupported_extension_uses_legacy_writer_parameters(extension: str) -> None:
    """Bypass export preparation and its dialog for writable legacy formats."""
    obj = SimpleNamespace(data=np.array([[1, 3]], dtype=np.float32))
    panel = SimpleNamespace(parentWidget=Mock())

    with patch.object(ImageExportGUIParam, "edit") as edit:
        result = ImagePanel.edit_export_parameters(panel, obj, f"image.{extension}")

    assert result == (True, None)
    edit.assert_not_called()
    panel.parentWidget.assert_not_called()


def test_export_dialog_validation_error_is_not_suppressed() -> None:
    """Propagate validation errors raised after GUI parameter creation."""
    obj = SimpleNamespace(data=np.array([[1, 3]], dtype=np.float32))
    panel = SimpleNamespace(parentWidget=Mock(return_value=None))
    guiparam = Mock()
    guiparam.edit.side_effect = ValueError("invalid parameters")

    with (
        patch.object(
            image_panel, "create_image_export_gui_param", return_value=guiparam
        ),
        pytest.raises(ValueError, match="invalid parameters"),
    ):
        ImagePanel.edit_export_parameters(panel, obj, "image.png")


def test_explicit_image_export_parameter_is_forwarded() -> None:
    """Forward an explicit export parameter through the image writer hook."""
    obj = object()
    param = ImageExportParam.create(normalization="minmax")
    with patch.object(image_panel, "write_image") as write_image:
        ImagePanel.write_object_to_file(SimpleNamespace(), obj, "image.png", param)
    write_image.assert_called_once_with("image.png", obj, param)


def test_default_export_hooks_preserve_signal_registry_write() -> None:
    """Keep the generic signal/default path parameter-free and registry-based."""
    obj = object()
    registry = SimpleNamespace(write=Mock())
    panel = SimpleNamespace(IO_REGISTRY=registry)

    assert BaseDataPanel.edit_export_parameters(panel, obj, "signal.csv") == (
        True,
        None,
    )
    BaseDataPanel.write_object_to_file(panel, obj, "signal.csv")
    registry.write.assert_called_once_with("signal.csv", obj)


def test_history_codec_preserves_aligned_optional_export_parameters() -> None:
    """Round-trip mixed export parameter lists without losing alignment."""
    param = ImageExportParam.create(normalization="minmax", target_dtype="auto")

    decoded = decode_kwargs(encode_kwargs({"export_params": [param, None]}))

    decoded_params = decoded["export_params"]
    assert isinstance(decoded_params[0], ImageExportParam)
    assert decoded_params[0].normalization == "minmax"
    assert decoded_params[0].target_dtype == "auto"
    assert decoded_params[1] is None


def test_history_hdf5_preserves_aligned_optional_export_parameters(tmp_path) -> None:
    """Round-trip mixed export parameters through the native HDF5 codec."""
    param = ImageExportParam.create(normalization="minmax", target_dtype="auto")
    action = HistoryAction(
        title="Export",
        target="imagepanel",
        method_name="save_to_files",
        kwargs={"export_params": [param, None]},
    )
    session = HistorySession(number=1)
    session.add_action(action)
    path = str(tmp_path / "history.dlhist")

    with NativeH5Writer(path) as writer:
        writer.write_object_list([session], "history_session")
    with NativeH5Reader(path) as reader:
        loaded = reader.read_object_list("history_session", HistorySession)[0]

    decoded_params = loaded.actions[0].kwargs["export_params"]
    assert isinstance(decoded_params[0], ImageExportParam)
    assert decoded_params[0].normalization == "minmax"
    assert decoded_params[0].target_dtype == "auto"
    assert decoded_params[1] is None


@pytest.mark.parametrize("legacy_none", [None, ""])
def test_history_codec_decodes_legacy_optional_dataset_entries(legacy_none) -> None:
    """Decode legacy Python and HDF5 representations of optional entries."""
    param = ImageExportParam.create(normalization="minmax")
    encoded = encode_kwargs({"export_params": [param, None]})
    encoded["export_params"]["__dataset_list_json__"][1] = legacy_none

    decoded_params = decode_kwargs(encoded)["export_params"]

    assert isinstance(decoded_params[0], ImageExportParam)
    assert decoded_params[1] is None


def test_history_codec_malformed_dataset_list_payload_is_safe() -> None:
    """Keep the warning and empty-list fallback for malformed nonempty JSON."""
    encoded = {"export_params": {"__dataset_list_json__": ["not-json"]}}

    with pytest.warns(UserWarning, match="DataSet-list"):
        decoded = decode_kwargs(encoded)

    assert decoded["export_params"] == []


def test_history_codec_preserves_primitive_lists() -> None:
    """Round-trip primitive lists without treating them as DataSet lists."""
    kwargs = {
        "filenames": ["first.png", "second.png"],
        "object_uuids": ["1", "2"],
    }

    assert decode_kwargs(encode_kwargs(kwargs)) == kwargs


def test_history_codec_preserves_homogeneous_dataset_lists() -> None:
    """Round-trip lists containing only export parameter DataSets."""
    params = [
        ImageExportParam.create(normalization="none"),
        ImageExportParam.create(normalization="minmax"),
    ]

    decoded = decode_kwargs(encode_kwargs({"export_params": params}))

    decoded_params = decoded["export_params"]
    assert all(isinstance(param, ImageExportParam) for param in decoded_params)
    assert [param.normalization for param in decoded_params] == ["none", "minmax"]
