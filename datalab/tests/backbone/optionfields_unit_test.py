# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the configuration option fields used by DataLab
(:mod:`sigimax.config`).
"""

import os.path as osp

import guidata.dataset as gds
import pytest
from sigimax.config import (
    AppOptionsContainer,
    ConfigPathOptionField,
    DataSetOptionField,
    FontOptionField,
    WorkingDirOptionField,
)

# Ensure the INI configuration backend (CONF) is initialized with an application
# name so that Configuration.get_path resolves inside the config directory.
import datalab.config  # noqa: F401  # pylint: disable=unused-import


class _StubContainer(AppOptionsContainer):
    """Minimal options container for testing individual option fields."""


class _SampleParam(gds.DataSet):
    """Simple DataSet used to exercise DataSetOptionField."""

    value = gds.IntItem("Value", default=3)


def test_config_path_option_field() -> None:
    """ConfigPathOptionField resolves basenames and round-trips raw values."""
    container = _StubContainer()
    field = ConfigPathOptionField(container, "traceback_log_path", ".DataLab_tb.log")

    resolved = field.get()
    assert osp.basename(resolved) == ".DataLab_tb.log"
    assert osp.isabs(resolved)

    # Storage accessors expose the bare basename (no path resolution).
    assert field.to_storage() == ".DataLab_tb.log"
    field.from_storage(".Other.log")
    assert field.to_storage() == ".Other.log"
    assert osp.basename(field.get()) == ".Other.log"

    # A full path (not a bare basename) is rejected on get().
    field.from_storage(osp.join("sub", "dir", "file.log"))
    with pytest.raises(ValueError):
        field.get()


def test_working_dir_option_field(tmp_path) -> None:
    """WorkingDirOptionField validates directories and tolerates missing ones."""
    container = _StubContainer()
    field = WorkingDirOptionField(container, "base_dir", "")

    # Setting an existing directory stores it and get() returns it.
    field.set(str(tmp_path))
    assert field.get() == str(tmp_path)

    # Setting a file path stores its parent directory.
    a_file = tmp_path / "data.txt"
    a_file.write_text("x", encoding="utf-8")
    field.set(str(a_file))
    assert field.get() == str(tmp_path)

    # Setting an invalid directory raises.
    with pytest.raises(FileNotFoundError):
        field.set(str(tmp_path / "does_not_exist" / "child"))

    # get() returns "" when the stored directory no longer exists, but the raw
    # value is preserved.
    missing = str(tmp_path / "gone")
    field.from_storage(missing)
    assert field.get() == ""
    assert field.to_storage() == missing


def test_font_option_field() -> None:
    """FontOptionField converts lists to tuples and builds a QFont."""
    container = _StubContainer()
    field = FontOptionField(container, "small_mono_font", ("Consolas", 8, False))

    # Lists are normalized to tuples on set.
    field.set(["Courier New", 10, True])
    assert field.get() == ("Courier New", 10, True)

    # get_font returns a QFont matching the stored specification.
    from qtpy.QtWidgets import QApplication  # pylint: disable=import-outside-toplevel

    _app = QApplication.instance() or QApplication([])
    font = field.get_font()
    assert font.family() == "Courier New"
    assert font.pointSize() == 10
    assert font.bold() is True


def test_dataset_option_field() -> None:
    """DataSetOptionField falls back to the default instance and round-trips JSON."""
    container = _StubContainer()
    default = _SampleParam()
    default.value = 7
    field = DataSetOptionField(container, "sample_param", default_instance=default)

    # Without an explicit value, get() returns the default instance.
    assert field.get() is default
    assert field.get_raw() is None
    assert field.to_json() is None

    # Setting an explicit value takes precedence.
    param = _SampleParam()
    param.value = 42
    field.set(param)
    assert field.get() is param
    assert field.get_raw() is param

    # JSON round-trip restores the stored value.
    json_str = field.to_json()
    assert json_str is not None
    field.from_storage(None)
    field.from_json(json_str)
    assert field.get().value == 42

    # set_default_instance updates the fallback used when no value is set.
    field.from_storage(None)
    new_default = _SampleParam()
    new_default.value = 99
    field.set_default_instance(new_default)
    assert field.get() is new_default


def test_dataset_option_field_invalid_json_uses_default() -> None:
    """An unresolved DataSet class is discarded and falls back to the default."""
    container = _StubContainer()
    default = _SampleParam()
    field = DataSetOptionField(container, "sample_param", default_instance=default)
    field.from_json('{"class_module": "missing_module", "class_name": "MissingParam"}')

    assert field.get() is default
    assert not container.is_option_initialized("sample_param")
