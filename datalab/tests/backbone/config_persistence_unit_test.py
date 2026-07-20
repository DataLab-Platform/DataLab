# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the DataLab configuration persistence layer
(:mod:`datalab.config_persistence`).
"""

import guidata.dataset as gds
from sigimax.utils.conf import AppUserConfig

from datalab.config.config_options import DataLabOptions
from datalab.config.config_persistence import (
    get_ini_location,
    get_uncategorized_fields,
    has_persisted_option,
    load_options_from_ini,
    remove_persisted_option,
    save_options_to_ini,
)


class _SampleParam(gds.DataSet):
    """Simple DataSet used to exercise DataSet options."""

    value = gds.IntItem("Value", default=1)


def _make_conf() -> AppUserConfig:
    """Return an isolated in-memory UserConfig backend for tests."""
    conf = AppUserConfig({})
    conf.set_application("DataLab_pytest", "1.0.0", load=False)
    return conf


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


def test_font_uses_three_ini_keys() -> None:
    """Font options are stored as three separate INI keys."""
    conf = _make_conf()
    src = DataLabOptions()
    src.small_mono_font.set(("Arial", 12, True))

    save_options_to_ini(src, conf, save=False)

    assert conf.get("proc", "small_mono_font_family") == "Arial"
    assert conf.get("proc", "small_mono_font_size") == 12
    assert conf.get("proc", "small_mono_font_bold") is True

    dst = DataLabOptions()
    load_options_from_ini(dst, conf)
    assert dst.small_mono_font.get() == ("Arial", 12, True)


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
