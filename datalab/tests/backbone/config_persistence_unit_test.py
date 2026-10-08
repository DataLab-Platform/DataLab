# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the DataLab configuration persistence layer
(:mod:`datalab.config.persistence`).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import guidata.dataset as gds
import pytest
from sigimax.utils import conf as confmod
from sigimax.utils.conf import AppUserConfig

from datalab.config.core import DataLabShapeParam
from datalab.config.options import DataLabOptions
from datalab.config.persistence import (
    CONF_VERSION,
    DataLabUserConfig,
    atomic_save_configuration,
    get_ini_location,
    get_uncategorized_fields,
    has_persisted_option,
    load_options_from_ini,
    migrate_legacy_configuration,
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


def test_legacy_configuration_migrates_without_modifying_source(
    tmp_path, monkeypatch
) -> None:
    """The first typed startup copies legacy values and preserves downgrade."""
    fixture = Path(__file__).parents[2] / "data" / "tests" / "config" / "DataLab_v1.ini"
    fixture_bytes = fixture.read_bytes()
    legacy_filename = tmp_path / "DataLab_v1.ini"
    shutil.copyfile(fixture, legacy_filename)
    legacy_bytes = legacy_filename.read_bytes()

    typed = DataLabUserConfig({})
    monkeypatch.setattr(typed, "get_path", lambda basename: str(tmp_path / basename))
    typed.set_application("DataLab_v1", CONF_VERSION, load=False)
    monkeypatch.setattr(confmod, "CONF", typed)
    options = DataLabOptions()

    assert migrate_legacy_configuration(options, str(legacy_filename), typed)
    assert fixture.read_bytes() == fixture_bytes
    assert legacy_filename.read_bytes() == legacy_bytes
    assert typed.filename().endswith("DataLab_v1_typed.ini")
    assert options.rpc_server_enabled.get() is False
    assert options.rpc_server_port.get() is None
    assert options.max_shapes_to_draw.get() == 1000
    assert options.plugins_path.get() == r"C:\Users\anonymous\.DataLab_v1\plugins"
    assert options.plugins_path_list.get() == [
        r"C:\Users\anonymous\.DataLab_v1\plugins"
    ]
    assert options.plugins_enabled_list.get() is None
    assert isinstance(options.sig_shape_param.get(), DataLabShapeParam)

    reloaded_backend = DataLabUserConfig({})
    monkeypatch.setattr(
        reloaded_backend, "get_path", lambda basename: str(tmp_path / basename)
    )
    reloaded_backend.set_application("DataLab_v1", CONF_VERSION, load=True)
    reloaded_options = DataLabOptions()
    load_options_from_ini(reloaded_options, reloaded_backend)
    assert reloaded_options.rpc_server_enabled.get() is False
    assert reloaded_options.rpc_server_port.get() is None
    assert reloaded_options.max_shapes_to_draw.get() == 1000
    assert reloaded_options.plugins_path_list.get() == [
        r"C:\Users\anonymous\.DataLab_v1\plugins"
    ]
    assert reloaded_options.plugins_enabled_list.get() is None
    assert isinstance(reloaded_options.sig_shape_param.get(), DataLabShapeParam)
    assert not migrate_legacy_configuration(options, str(legacy_filename), typed)


def test_atomic_configuration_save_cleans_up_after_replace_error(
    tmp_path, monkeypatch
) -> None:
    """A failed atomic replacement leaves no typed or temporary file."""
    config = DataLabUserConfig({})
    monkeypatch.setattr(config, "get_path", lambda basename: str(tmp_path / basename))
    config.set_application("DataLab_v1", CONF_VERSION, load=False)
    config.set("main", "color_mode", "dark", save=False)

    def raise_replace_error(_source, _destination) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("datalab.config.persistence.os.replace", raise_replace_error)

    with pytest.raises(OSError, match="replace failed"):
        atomic_save_configuration(config)

    assert not (tmp_path / "DataLab_v1_typed.ini").exists()
    assert not list(tmp_path.glob("*.tmp"))


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
    src.plugins_path_list.set(["/tmp/plugins_a", "/tmp/plugins_b"])
    src.plugins_enabled_list.set(["Test Plugin 1"])
    src.macro_console_max_lines.set(4242)
    src.ai_temperature.set(0.9)

    # Raw fields (config path / working directory)
    src.traceback_log_path.from_storage(".DataLab_custom.log")

    save_options_to_ini(src, conf, save=False)

    dst = DataLabOptions()
    load_options_from_ini(dst, conf)

    assert dst.process_isolation_enabled.get() is False
    assert dst.available_memory_threshold.get() == 750
    assert dst.ai_provider.get() == "local"
    assert dst.operation_mode.get() == "pairwise"
    assert dst.window_size.get() == (1234, 567)
    assert dst.plugins_path_list.get() == ["/tmp/plugins_a", "/tmp/plugins_b"]
    assert dst.plugins_enabled_list.get() == ["Test Plugin 1"]
    assert dst.macro_console_max_lines.get() == 4242
    assert abs(dst.ai_temperature.get() - 0.9) < 1e-9
    assert dst.traceback_log_path.to_storage() == ".DataLab_custom.log"


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


def test_invalid_dataset_option_is_removed_on_load() -> None:
    """An unresolved persisted DataSet is discarded from the INI backend."""
    conf = _make_conf()
    options = DataLabOptions()
    section, ini_key = get_ini_location(options, "sig_shape_param")
    conf.set(
        section,
        ini_key,
        '{"class_module": "missing_module", "class_name": "MissingParam"}',
        save=False,
    )

    load_options_from_ini(options, conf)

    assert not conf.has_option(section, ini_key)
    assert options.sig_shape_param.get() is None
    assert not options.is_option_initialized("sig_shape_param")


def test_config_app_name_is_bound_at_import_time(tmp_path) -> None:
    """Import-time consumers resolve paths under the application directory.

    ``datalab.plugins`` resolves its default path when imported, which happens long
    before ``initialize()`` runs: an unnamed backend would silently relocate user
    plugins to ``~/.none/plugins``. A fresh interpreter is required, as the import
    order cannot be reproduced in the current one.
    """
    env = dict(
        os.environ,
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),
        XDG_CONFIG_HOME=str(tmp_path),
        QT_QPA_PLATFORM="offscreen",
    )
    code = (
        "import datalab.plugins;"
        "from datalab.config.appinfo import get_config_app_name;"
        "print(datalab.plugins.PLUGINS_DEFAULT_PATH);"
        "print(get_config_app_name())"
    )
    stdout = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    plugin_path, app_name = stdout.splitlines()[-2:]

    assert Path(plugin_path).parent.name == f".{app_name}"
