# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Parameterized RAW image import unit tests."""

from __future__ import annotations

import os.path as osp
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from sigima.io import RawImageImportParam
from sigima.io.image import ImageIORegistry
from sigima.objects import create_image

from datalab.adapters_plotpy import create_adapter_from_object
from datalab.gui.panel import base as panel_base
from datalab.gui.panel import image as panel_image
from datalab.gui.panel.base import BaseDataPanel
from datalab.gui.panel.image import ImagePanel
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.history.action import HistoryAction
from datalab.history.session import HistorySession


class HistoryPanelStub:
    """Minimal history collaborator used by the generic import flow."""

    def __init__(self) -> None:
        self.actions: list[HistoryAction] = []
        self.tree = SimpleNamespace(refresh_action_item=Mock())

    def maybe_start_session_for_input(self, *, load: bool = False) -> bool:
        """Accept the current history session."""
        del load
        return False

    def add_ui_entry(
        self,
        title: str,
        target: str,
        method_name: str,
        save_state: bool = True,
        **kwargs,
    ) -> HistoryAction:
        """Record and return one UI action."""
        del save_state
        action = HistoryAction(
            title=title,
            target=target,
            method_name=method_name,
            kwargs=kwargs,
        )
        self.actions.append(action)
        return action

    @staticmethod
    def session_prompt_suppressed():
        """Return a no-op batch context."""
        return nullcontext()

    @staticmethod
    def capture_outputs(_action):
        """Return a no-op output capture context."""
        return nullcontext()

    @staticmethod
    def is_output_suppressed() -> bool:
        """Return whether replay output is suppressed."""
        return False


class RawLoadPanelStub:
    """Panel stub retaining the production import orchestration hooks."""

    PANEL_STR_ID = "image"
    load_from_files = BaseDataPanel.load_from_files
    load_from_directory = BaseDataPanel.load_from_directory
    edit_import_parameters = ImagePanel.edit_import_parameters
    edit_raw_import_parameters = ImagePanel.edit_raw_import_parameters
    apply_import_parameters = ImagePanel.apply_import_parameters

    def __init__(self) -> None:
        self.historypanel = HistoryPanelStub()
        self.mainwindow = SimpleNamespace(
            confirm_memory_state=Mock(return_value=True),
            historypanel=self.historypanel,
            imagepanel=self,
        )
        self.loaded_objects = [object()]
        self.load_file = Mock(return_value=self.loaded_objects)
        self.add_group = Mock(return_value=SimpleNamespace())
        self.add_object = Mock()
        setattr(self, "parentWidget", Mock(return_value=None))
        setattr(self, "_BaseDataPanel__load_from_file", self.load_file)

    def loaded_params(self) -> dict[str, RawImageImportParam | None]:
        """Return the import parameter used for each loaded file."""
        return {
            item.args[0]: item.kwargs["import_param"]
            for item in self.load_file.call_args_list
        }


def run_stub_import(panel: RawLoadPanelStub, filenames, **kwargs):
    """Run the generic import flow without Qt error dialogs."""
    with (
        patch.object(
            panel_base,
            "qt_try_loadsave_file",
            side_effect=lambda *_args: nullcontext(),
        ),
        patch.object(panel_base.Conf.base_dir, "set"),
    ):
        return panel.load_from_files(filenames, **kwargs)


def run_stub_directory_import(panel: RawLoadPanelStub, directory, **kwargs):
    """Run the directory import flow without Qt dialogs or progress bars."""
    progress = SimpleNamespace(setValue=Mock(), wasCanceled=Mock(return_value=False))
    with (
        patch.object(
            panel_base,
            "qt_try_loadsave_file",
            side_effect=lambda *_args: nullcontext(),
        ),
        patch.object(panel_base.Conf.base_dir, "set"),
        patch.object(
            panel_base, "create_progress_bar", return_value=nullcontext(progress)
        ),
        patch.object(panel_base, "get_uuid", return_value="group-uuid"),
    ):
        return panel.load_from_directory(str(directory), **kwargs)


@contextmanager
def interactive_question(answer: int):
    """Answer the multi-file RAW question with ``answer`` in interactive mode."""
    with (
        patch.object(panel_image, "execenv", SimpleNamespace(unattended=False)),
        patch.object(
            panel_image.QW.QMessageBox, "question", return_value=answer
        ) as question,
    ):
        yield question


def confirm_raw_dialog(**values):
    """Return a dialog edit replacement that sets values and confirms."""

    def edit(param, parent=None):
        del parent
        for name, value in values.items():
            setattr(param, name, value)
        return True

    return edit


def create_files(directory, *names: str) -> list[str]:
    """Create empty files and return their paths."""
    paths = []
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        paths.append(str(path))
    return paths


def test_raw_dialog_confirmation_forwards_backend_parameters(tmp_path) -> None:
    """Forward confirmed RAW settings to the registry-facing load helper."""
    filename = str(tmp_path / "image.raw")
    panel = RawLoadPanelStub()

    with patch.object(
        RawImageImportParam,
        "edit",
        autospec=True,
        side_effect=confirm_raw_dialog(
            dtype="uint8",
            width=3,
            height=2,
            offset=4,
            count=2,
            gap=1,
            little_endian=False,
            white_is_zero=True,
        ),
    ):
        run_stub_import(panel, [filename])

    import_param = panel.load_file.call_args.kwargs["import_param"]
    assert isinstance(import_param, RawImageImportParam)
    assert (
        import_param.dtype,
        import_param.width,
        import_param.height,
        import_param.offset,
        import_param.count,
        import_param.gap,
        import_param.little_endian,
        import_param.white_is_zero,
    ) == ("uint8", 3, 2, 4, 2, 1, False, True)
    history_param = panel.historypanel.actions[0].kwargs["import_params"][0]
    assert history_param is not import_param
    import_param.width = 9
    assert history_param.width == 3


def test_raw_dialog_cancellation_skips_read_and_history(tmp_path) -> None:
    """Do not read or record a RAW file after dialog cancellation."""
    panel = RawLoadPanelStub()

    with patch.object(RawImageImportParam, "edit", return_value=False):
        objects = run_stub_import(panel, [str(tmp_path / "cancelled.raw")])

    assert objects == []
    panel.load_file.assert_not_called()
    assert not panel.historypanel.actions


def test_non_raw_import_keeps_parameter_free_path(tmp_path) -> None:
    """Keep normal image imports parameter-free and dialog-free."""
    filename = str(tmp_path / "image.npy")
    panel = RawLoadPanelStub()

    with patch.object(RawImageImportParam, "edit") as edit:
        objects = run_stub_import(panel, [filename])

    assert objects == panel.loaded_objects
    edit.assert_not_called()
    panel.load_file.assert_called_once_with(
        filename,
        create_group=False,
        add_objects=True,
        import_param=None,
    )
    assert "import_params" not in panel.historypanel.actions[0].kwargs


def test_non_raw_duplicate_filenames_keep_existing_behavior(tmp_path) -> None:
    """Do not apply RAW expansion deduplication to ordinary imports."""
    filename = str(tmp_path / "image.npy")
    panel = RawLoadPanelStub()

    run_stub_import(panel, [filename, filename])

    assert panel.load_file.call_count == 2
    assert panel.historypanel.actions[0].kwargs["filenames"] == [filename, filename]


def test_several_raw_files_share_parameters_after_confirmation(tmp_path) -> None:
    """Apply one dialog to all RAW files when the user accepts shared parameters."""
    first, second, plain = create_files(tmp_path, "b.raw", "a.RAW", "c.npy")
    panel = RawLoadPanelStub()

    with (
        interactive_question(panel_image.QW.QMessageBox.Yes) as question,
        patch.object(
            RawImageImportParam,
            "edit",
            autospec=True,
            side_effect=confirm_raw_dialog(width=3, height=1),
        ) as edit,
    ):
        run_stub_import(panel, [first, plain, second])

    question.assert_called_once()
    assert edit.call_count == 1
    assert "2" in edit.call_args.args[0].get_title()
    params = panel.loaded_params()
    assert list(params) == [second, first, plain]
    assert params[plain] is None
    assert params[first] is not params[second]
    assert (params[first].width, params[second].width) == (3, 3)
    history_params = panel.historypanel.actions[0].kwargs["import_params"]
    assert [param.width if param else None for param in history_params] == [
        3,
        3,
        None,
    ]
    assert history_params[0] is not params[second]


def test_several_raw_files_get_one_dialog_each(tmp_path) -> None:
    """Edit each RAW file separately, skipping only the cancelled one."""
    first, second, third = create_files(tmp_path, "a.raw", "b.raw", "c.raw")
    panel = RawLoadPanelStub()
    titles = []
    answers = iter([(True, 3), (False, None), (True, 5)])

    def edit(param, parent=None):
        del parent
        titles.append(param.get_title())
        confirmed, width = next(answers)
        if confirmed:
            param.width = width
        return confirmed

    with (
        interactive_question(panel_image.QW.QMessageBox.No),
        patch.object(RawImageImportParam, "edit", autospec=True, side_effect=edit),
    ):
        run_stub_import(panel, [third, first, second])

    assert ["a.raw" in titles[0], "b.raw" in titles[1], "c.raw" in titles[2]] == [
        True,
        True,
        True,
    ]
    params = panel.loaded_params()
    assert list(params) == [first, third]
    assert (params[first].width, params[third].width) == (3, 5)
    assert panel.historypanel.actions[0].kwargs["filenames"] == [first, third]


def test_cancelling_several_raw_files_question_aborts_import(tmp_path) -> None:
    """Abort the whole import, including non-RAW files, from the question."""
    filenames = create_files(tmp_path, "a.raw", "b.raw", "c.npy")
    panel = RawLoadPanelStub()

    with (
        interactive_question(panel_image.QW.QMessageBox.Cancel),
        patch.object(RawImageImportParam, "edit") as edit,
    ):
        objects = run_stub_import(panel, filenames)

    assert objects == []
    edit.assert_not_called()
    panel.load_file.assert_not_called()
    assert not panel.historypanel.actions


def test_cancelling_shared_raw_dialog_keeps_other_files(tmp_path) -> None:
    """Skip all RAW files, but still load other files, on shared dialog cancel."""
    raw1, raw2, plain = create_files(tmp_path, "a.raw", "b.raw", "c.npy")
    panel = RawLoadPanelStub()

    with (
        interactive_question(panel_image.QW.QMessageBox.Yes),
        patch.object(RawImageImportParam, "edit", return_value=False),
    ):
        run_stub_import(panel, [raw1, raw2, plain])

    assert panel.loaded_params() == {plain: None}


def test_import_param_applies_to_raw_files_without_dialog(tmp_path) -> None:
    """Apply a caller-provided RAW parameter set (e.g. remote control)."""
    raw1, raw2, plain = create_files(tmp_path, "a.raw", "b.raw", "c.npy")
    param = RawImageImportParam.create(dtype="uint8", width=4, height=2)
    panel = RawLoadPanelStub()

    with (
        interactive_question(panel_image.QW.QMessageBox.Yes) as question,
        patch.object(RawImageImportParam, "edit") as edit,
    ):
        run_stub_import(panel, [plain, raw2, raw1], param=param)

    question.assert_not_called()
    edit.assert_not_called()
    params = panel.loaded_params()
    assert params[plain] is None
    assert params[raw1] is not param and params[raw1] is not params[raw2]
    assert (params[raw1].width, params[raw2].dtype) == (4, "uint8")
    history_params = panel.historypanel.actions[0].kwargs["import_params"]
    assert history_params[0].width == 4 and history_params[2] is None


def test_invalid_import_param_is_rejected(tmp_path) -> None:
    """Reject unsupported or conflicting import parameter arguments."""
    (filename,) = create_files(tmp_path, "a.raw")
    panel = RawLoadPanelStub()
    param = RawImageImportParam.create(width=2, height=2)

    with pytest.raises(TypeError, match="RawImageImportParam"):
        run_stub_import(panel, [filename], param=object())
    with pytest.raises(ValueError, match="cannot be combined"):
        run_stub_import(panel, [filename], param=param, import_params=[param])
    with pytest.raises(TypeError, match="does not support import parameters"):
        BaseDataPanel.edit_import_parameters(panel, [filename], param)
    panel.load_file.assert_not_called()


def test_directory_import_asks_raw_parameters_once_for_all_folders(
    tmp_path,
) -> None:
    """Ask RAW parameters once for a directory tree and record them per file."""
    root_raw, plain, sub_raw = create_files(tmp_path, "a.raw", "b.npy", "sub/c.raw")
    panel = RawLoadPanelStub()

    with (
        interactive_question(panel_image.QW.QMessageBox.Yes) as question,
        patch.object(
            RawImageImportParam,
            "edit",
            autospec=True,
            side_effect=confirm_raw_dialog(width=6, height=1),
        ) as edit,
    ):
        run_stub_directory_import(panel, tmp_path)

    question.assert_called_once()
    assert edit.call_count == 1
    params = panel.loaded_params()
    assert params[plain] is None
    assert (params[root_raw].width, params[sub_raw].width) == (6, 6)
    assert panel.add_group.call_count == 2
    (action,) = panel.historypanel.actions
    assert action.method_name == "load_from_directory"
    assert action.kwargs["import_filenames"] == ["a.raw", osp.join("sub", "c.raw")]
    assert [param.width for param in action.kwargs["import_params"]] == [6, 6]
    assert action.kwargs["import_params"][0] is not params[root_raw]


def test_directory_replay_reuses_recorded_raw_parameters(tmp_path) -> None:
    """Replay a directory import with recorded parameters, asking only new files."""
    recorded_raw, new_raw = create_files(tmp_path, "a.raw", "sub/new.raw")
    recorded = RawImageImportParam.create(width=2, height=3)
    panel = RawLoadPanelStub()

    with (
        interactive_question(panel_image.QW.QMessageBox.Yes) as question,
        patch.object(
            RawImageImportParam,
            "edit",
            autospec=True,
            side_effect=confirm_raw_dialog(width=7, height=1),
        ) as edit,
    ):
        run_stub_directory_import(
            panel,
            tmp_path,
            import_filenames=["a.raw"],
            import_params=[recorded],
        )

    question.assert_not_called()
    assert edit.call_count == 1
    assert "new.raw" in edit.call_args.args[0].get_title()
    params = panel.loaded_params()
    assert params[recorded_raw] is recorded
    assert params[new_raw].width == 7
    (action,) = panel.historypanel.actions
    assert [param.width for param in action.kwargs["import_params"]] == [2, 7]


def test_directory_import_without_raw_files_records_directory_only(
    tmp_path,
) -> None:
    """Keep directory history entries unchanged when no RAW file is imported."""
    create_files(tmp_path, "a.npy", "sub/b.npy")
    panel = RawLoadPanelStub()

    with patch.object(RawImageImportParam, "edit") as edit:
        run_stub_directory_import(panel, tmp_path)

    edit.assert_not_called()
    assert panel.load_file.call_count == 2
    (action,) = panel.historypanel.actions
    assert action.kwargs == {"directory": str(tmp_path)}


def test_partial_failure_keeps_successful_filenames_and_params_aligned(
    tmp_path,
) -> None:
    """Drop a failed file and only its matching parameter from history."""
    plain = str(tmp_path / "a.npy")
    failed = str(tmp_path / "m.raw")
    loaded = str(tmp_path / "z.raw")
    failed_param = RawImageImportParam.create(width=2, height=2)
    loaded_param = RawImageImportParam.create(width=3, height=1)
    caller_params = [loaded_param, None, failed_param]
    panel = RawLoadPanelStub()

    def load_file(filename, **_kwargs):
        if filename == failed:
            raise OSError("unreadable")
        return [object()]

    panel.load_file.side_effect = load_file
    run_stub_import(
        panel,
        [loaded, plain, failed],
        import_params=caller_params,
        ignore_errors=True,
    )

    action = panel.historypanel.actions[0]
    assert action.kwargs["filenames"] == [plain, loaded]
    history_param = action.kwargs["import_params"][1]
    assert action.kwargs["import_params"][0] is None
    assert history_param is not loaded_param
    assert panel.load_file.call_args_list == [
        call(plain, create_group=False, add_objects=True, import_param=None),
        call(
            failed,
            create_group=False,
            add_objects=True,
            import_param=failed_param,
        ),
        call(
            loaded,
            create_group=False,
            add_objects=True,
            import_param=loaded_param,
        ),
    ]
    loaded_param.width = 9
    caller_params.clear()
    assert action.kwargs["import_params"][0] is None
    assert history_param.width == 3


def test_history_replay_uses_recorded_params_without_dialog(tmp_path) -> None:
    """Replay a RAW load with its backend parameters and no dialog."""
    filename = str(tmp_path / "image.raw")
    param = RawImageImportParam.create(dtype="uint8", width=2, height=2)
    panel = RawLoadPanelStub()
    action = HistoryAction(
        title="RAW import",
        target="imagepanel",
        method_name="load_from_files",
        kwargs={"filenames": [filename], "import_params": [param]},
    )

    with (
        patch.object(RawImageImportParam, "edit") as edit,
        patch.object(
            panel_base,
            "qt_try_loadsave_file",
            side_effect=lambda *_args: nullcontext(),
        ),
        patch.object(panel_base.Conf.base_dir, "set"),
    ):
        action.replay_ui(panel.mainwindow, edit=False)

    edit.assert_not_called()
    panel.load_file.assert_called_once_with(
        filename,
        create_group=False,
        add_objects=True,
        import_param=param,
    )


def test_history_hdf5_preserves_optional_raw_import_parameters(tmp_path) -> None:
    """Round-trip mixed RAW import parameters through native HDF5 history."""
    param = RawImageImportParam.create(
        dtype="float32",
        width=4,
        height=3,
        offset=8,
        count=2,
        gap=4,
        little_endian=False,
        white_is_zero=True,
    )
    action = HistoryAction(
        title="RAW import",
        target="imagepanel",
        method_name="load_from_files",
        kwargs={"import_params": [param, None]},
    )
    session = HistorySession(number=1)
    session.add_action(action)
    path = str(tmp_path / "history.dlhist")

    with NativeH5Writer(path) as writer:
        writer.write_object_list([session], "history_session")
    with NativeH5Reader(path) as reader:
        loaded = reader.read_object_list("history_session", HistorySession)[0]

    decoded = loaded.actions[0].kwargs["import_params"]
    assert isinstance(decoded[0], RawImageImportParam)
    assert (
        decoded[0].dtype,
        decoded[0].width,
        decoded[0].height,
        decoded[0].offset,
        decoded[0].count,
        decoded[0].gap,
        decoded[0].little_endian,
        decoded[0].white_is_zero,
    ) == ("float32", 4, 3, 8, 2, 4, False, True)
    assert decoded[1] is None


def test_panel_load_helper_reads_raw_through_image_registry(tmp_path) -> None:
    """Read RAW bytes through the panel helper and the real image registry."""
    expected = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    filename = str(tmp_path / "image.raw")
    expected.tofile(filename)
    param = RawImageImportParam.create(width=2, height=2)
    panel = SimpleNamespace(
        IO_REGISTRY=ImageIORegistry,
        apply_import_parameters=Mock(),
        selection_changed=Mock(),
    )
    progress = SimpleNamespace(setValue=Mock(), wasCanceled=Mock(return_value=False))
    loader = getattr(BaseDataPanel, "_BaseDataPanel__load_from_file")

    with (
        patch.object(
            panel_base, "CallbackWorker", side_effect=lambda callback: callback
        ),
        patch.object(
            panel_base,
            "qt_long_callback",
            side_effect=lambda _panel, _title, callback, _show: callback(None),
        ),
        patch.object(
            panel_base,
            "create_progress_bar",
            return_value=nullcontext(progress),
        ),
    ):
        objects = loader(
            panel,
            filename,
            create_group=False,
            add_objects=False,
            import_param=param,
        )

    np.testing.assert_array_equal(objects[0].data, expected)
    panel.apply_import_parameters.assert_called_once_with(objects, param)


def test_white_is_zero_uses_inverted_gray_metadata_without_changing_data() -> None:
    """Represent white-is-zero through stable PlotPy options, not data inversion."""
    data = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    image = create_image("RAW", data=data.copy(), metadata={"white_is_zero": True})
    param = RawImageImportParam.create(
        dtype="uint8", width=2, height=2, white_is_zero=True
    )

    ImagePanel.apply_import_parameters(SimpleNamespace(), [image], param)

    np.testing.assert_array_equal(image.data, data)
    assert image.metadata["white_is_zero"] is True
    assert image.get_metadata_option("colormap") == "gray"
    assert image.get_metadata_option("invert_colormap") is True
    item = create_adapter_from_object(image).make_item()
    assert item.param.colormap == "gray"
    assert item.param.invert_colormap is True
