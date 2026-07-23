# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the DataLab configuration persistence layer
(:mod:`datalab.config_persistence`).
"""

import guidata.dataset as gds
import pytest
from sigimax.utils import conf as confmod
from sigimax.utils.conf import AppUserConfig

from datalab.config.config import (
    CONF_VERSION,
    DataLabShapeParam,
    DataLabUserConfig,
    LegacyConfigSnapshot,
    atomic_save_configuration,
    migrate_legacy_configuration,
)
from datalab.config.config_options import DataLabOptions
from datalab.config.config_persistence import (
    get_ini_location,
    get_uncategorized_fields,
    has_persisted_option,
    load_options_from_ini,
    remove_persisted_option,
    save_options_to_ini,
    save_runtime_option,
)


class _SampleParam(gds.DataSet):
    """Simple DataSet used to exercise DataSet options."""

    value = gds.IntItem("Value", default=1)


def _make_conf() -> AppUserConfig:
    """Return an isolated in-memory UserConfig backend for tests."""
    conf = AppUserConfig({})
    conf.set_application("DataLab_pytest", "1.0.0", load=False)
    return conf


class _DirectoryConfig(DataLabUserConfig):
    """DataLab config backend rooted in a temporary test directory."""

    def __init__(self, directory) -> None:
        self._directory = directory
        super().__init__({})

    def get_path(self, basename: str) -> str:
        """Return a path in the temporary test directory."""
        return str(self._directory / basename)


class _DirectoryLegacyConfig(LegacyConfigSnapshot):
    """Read-only legacy backend rooted in a temporary test directory."""

    def __init__(self, directory) -> None:
        self._directory = directory
        super().__init__("DataLab_v1")

    def get_path(self, basename: str) -> str:
        """Return a path in the temporary test directory."""
        return str(self._directory / basename)


def test_legacy_configuration_migrates_without_modifying_source(tmp_path) -> None:
    """The first typed startup copies legacy values and preserves downgrade."""
    legacy_filename = tmp_path / "DataLab_v1.ini"
    legacy = AppUserConfig({})
    legacy.name = "DataLab_v1"
    shape_json = gds.dataset_to_json(DataLabShapeParam()).replace(
        '"class_module": "datalab.config.config"',
        '"class_module": "datalab.config"',
    )
    legacy.read_dict(
        {
            "main": {
                "version": CONF_VERSION,
                "plugins_enabled_list": repr(["plugin_x"]),
                "plugins_path": repr(str(tmp_path / "legacy_plugins")),
            },
            "view": {
                "max_shapes_to_draw": "321",
                "sig_shape_param": shape_json,
            },
        }
    )
    with legacy_filename.open("w", encoding="utf-8") as stream:
        legacy.write(stream)
    legacy_bytes = legacy_filename.read_bytes()

    typed = _DirectoryConfig(tmp_path)
    typed.set_application("DataLab_v1", CONF_VERSION, load=False)
    options = DataLabOptions()

    assert migrate_legacy_configuration(options, str(legacy_filename), typed)
    assert legacy_filename.read_bytes() == legacy_bytes
    assert typed.filename().endswith("DataLab_v1_typed.ini")
    assert options.max_shapes_to_draw.get() == 321
    assert options.plugins_path.get() == str(tmp_path / "legacy_plugins")
    assert options.plugins_path_list.get() == [str(tmp_path / "legacy_plugins")]
    assert options.plugins_enabled_list.get() == ["plugin_x"]
    assert isinstance(options.sig_shape_param.get_raw(), DataLabShapeParam)

    reloaded_backend = _DirectoryConfig(tmp_path)
    reloaded_backend.set_application("DataLab_v1", CONF_VERSION, load=True)
    reloaded_options = DataLabOptions()
    load_options_from_ini(reloaded_options, reloaded_backend)
    assert reloaded_options.max_shapes_to_draw.get() == 321
    assert reloaded_options.plugins_path_list.get() == [
        str(tmp_path / "legacy_plugins")
    ]
    assert reloaded_options.plugins_enabled_list.get() == ["plugin_x"]
    assert isinstance(reloaded_options.sig_shape_param.get_raw(), DataLabShapeParam)
    assert not migrate_legacy_configuration(options, str(legacy_filename), typed)


def test_atomic_configuration_save_cleans_up_after_replace_error(
    tmp_path, monkeypatch
) -> None:
    """A failed atomic replacement leaves no typed or temporary file."""
    config = _DirectoryConfig(tmp_path)
    config.set_application("DataLab_v1", CONF_VERSION, load=False)
    config.set("main", "color_mode", "dark", save=False)

    def raise_replace_error(_source, _destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("datalab.config.config.os.replace", raise_replace_error)

    with pytest.raises(OSError, match="replace failed"):
        atomic_save_configuration(config)

    assert not (tmp_path / "DataLab_v1_typed.ini").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_legacy_backend_is_read_only(tmp_path) -> None:
    """DataLab 1.3 development mode cannot overwrite the DataLab 1.2 INI."""
    legacy_filename = tmp_path / "DataLab_v1.ini"
    legacy_filename.write_text("[main]\nversion = 1.0.0\n", encoding="utf-8")
    original_bytes = legacy_filename.read_bytes()
    backend = _DirectoryLegacyConfig(tmp_path)
    backend.set_application("DataLab_v1", CONF_VERSION, load=True)

    backend.set("main", "plugins_enabled", False)
    backend.remove_option("main", "version")
    backend.save()
    backend.cleanup()

    assert legacy_filename.read_bytes() == original_bytes


def test_section_map_is_complete() -> None:
    """Every option field is either categorized or explicitly not persisted."""
    opt = DataLabOptions()
    assert get_uncategorized_fields(opt) == []


def test_persisted_option_presence_and_removal() -> None:
    """Flat field names map to the expected INI option for removal."""
    conf = _make_conf()
    options = DataLabOptions()
    conf.set("macro", "console_max_lines", 123, save=False)

    assert has_persisted_option(options, "macro_console_max_lines", conf)
    assert remove_persisted_option(options, "macro_console_max_lines", conf)
    assert not has_persisted_option(options, "macro_console_max_lines", conf)
    assert not remove_persisted_option(options, "macro_console_max_lines", conf)


def test_uncategorized_option_has_no_persisted_location() -> None:
    """Non-persisted application metadata cannot reach the INI backend."""
    conf = _make_conf()
    options = DataLabOptions()

    assert not has_persisted_option(options, "app_name", conf)
    assert not remove_persisted_option(options, "app_name", conf)


def test_round_trip_across_types() -> None:
    """Values of various types round-trip through the INI backend."""
    conf = _make_conf()
    src = DataLabOptions()

    # bool / int / str / enum / tuple / list / prefixed sections
    src.process_isolation_enabled.set(False)
    src.available_memory_threshold.set(750)
    src.ai_provider.set("local")
    src.operation_mode.set("pairwise")
    src.window_size.set((1234, 567))
    src.plugins_path_list.set(["/tmp/a", "/tmp/b"])
    src.plugins_enabled_list.set(["plugin_x"])
    src.macro_console_max_lines.set(4242)
    src.ai_temperature.set(0.9)

    # Raw fields (config path / working directory)
    src.traceback_log_path.set_raw(".DataLab_custom.log")

    save_options_to_ini(src, conf, save=False)

    dst = DataLabOptions()
    load_options_from_ini(dst, conf)

    assert dst.process_isolation_enabled.get() is False
    assert dst.available_memory_threshold.get() == 750
    assert dst.ai_provider.get() == "local"
    assert dst.operation_mode.get() == "pairwise"
    assert dst.window_size.get() == (1234, 567)
    assert dst.plugins_path_list.get() == ["/tmp/a", "/tmp/b"]
    assert dst.plugins_enabled_list.get() == ["plugin_x"]
    assert dst.macro_console_max_lines.get() == 4242
    assert abs(dst.ai_temperature.get() - 0.9) < 1e-9
    assert dst.traceback_log_path.get_raw() == ".DataLab_custom.log"


def test_runtime_option_is_not_clobbered_by_bulk_save() -> None:
    """The XML-RPC port is persisted only through its single-key writer."""
    conf = _make_conf()
    options = DataLabOptions()
    options.rpc_server_port.set(12345)

    save_runtime_option(options, "rpc_server_port", conf)
    assert conf.get("main", "rpc_server_port") == 12345

    options.rpc_server_port.set(54321)
    options.color_mode.set("dark")
    save_options_to_ini(options, conf, save=False)

    assert conf.get("main", "rpc_server_port") == 12345
    assert conf.get("main", "color_mode") == "dark"


def test_datetime_is_escaped_in_ini_but_clean_in_memory() -> None:
    """Datetime formats are stored escaped (%%) but kept clean (%) in memory."""
    conf = _make_conf()
    src = DataLabOptions()
    src.sig_datetime_format_s.set("%H:%M:%S")

    save_options_to_ini(src, conf, save=False)

    # Stored value is percent-escaped for ConfigParser.
    section, ini_key = get_ini_location(src, "sig_datetime_format_s")
    raw_stored = conf.get(section, ini_key, raw=True)
    assert "%%" in raw_stored

    # Loaded value is back to the clean form.
    dst = DataLabOptions()
    load_options_from_ini(dst, conf)
    assert dst.sig_datetime_format_s.get() == "%H:%M:%S"


def test_font_uses_three_ini_keys(monkeypatch) -> None:
    """Font options are stored as three separate INI keys."""
    conf = _make_conf()
    monkeypatch.setattr(confmod, "CONF", conf)
    src = DataLabOptions()
    src.small_mono_font.set(("Arial", 12, True))

    save_options_to_ini(src, conf, save=False)

    assert conf.get("proc", "small_mono_font_family") == "Arial"
    assert conf.get("proc", "small_mono_font_size") == 12
    assert conf.get("proc", "small_mono_font_bold") is True
    assert has_persisted_option(src, "small_mono_font", conf)

    dst = DataLabOptions()
    load_options_from_ini(dst, conf)
    assert dst.small_mono_font.get() == ("Arial", 12, True)
    assert dst.small_mono_font.get(("Consolas", 8, False)) == (
        "Arial",
        12,
        True,
    )

    assert remove_persisted_option(dst, "small_mono_font", conf)
    assert not has_persisted_option(dst, "small_mono_font", conf)


def test_dataset_option_round_trip() -> None:
    """DataSet options round-trip through JSON in the INI backend."""
    conf = _make_conf()
    src = DataLabOptions()
    param = _SampleParam()
    param.value = 77
    src.sig_shape_param.set(param)

    save_options_to_ini(src, conf, save=False)

    dst = DataLabOptions()
    load_options_from_ini(dst, conf)
    assert dst.sig_shape_param.get().value == 77
