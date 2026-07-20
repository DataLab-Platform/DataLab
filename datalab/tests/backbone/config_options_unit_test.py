# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the DataLab options container
(:mod:`datalab.config_options`).
"""

import json
import os

import guidata.dataset as gds
from sigimax.config import get_conf
from sigimax.utils import conf as confmod
from sigimax.utils.conf import AppUserConfig

from datalab.config.config_options import DataLabOptions
from datalab.config.optionfields import (
    ConfigPathOptionField,
    DataSetOptionField,
    WorkingDirOptionField,
)


class _SampleParam(gds.DataSet):
    """Simple DataSet used to exercise DataSet options."""

    value = gds.IntItem("Value", default=1)


def _make_conf() -> AppUserConfig:
    """Return an isolated in-memory UserConfig backend for tests."""
    conf = AppUserConfig({})
    conf.set_application("DataLab_options_pytest", "1.0.0", load=False)
    return conf


def test_field_get_default_initializes_ini_and_json(monkeypatch) -> None:
    """A missing option default updates both persistence representations."""
    backend = _make_conf()
    monkeypatch.setattr(confmod, "CONF", backend)
    options = DataLabOptions()
    from datalab.config.config_persistence import (  # pylint: disable=import-outside-toplevel
        load_options_from_ini,
    )

    load_options_from_ini(options, backend)
    options.set_ini_persist_enabled(True)
    default = [900, 700]

    assert options.window_size.get(default) is default
    assert options.window_size.get() == (900, 700)
    assert backend.get("main", "window_size") == (900, 700)
    assert json.loads(os.environ[options.ENV_VAR])["window_size"] == [900, 700]


def test_field_get_default_preserves_existing_value(monkeypatch) -> None:
    """An existing INI value is returned without being overwritten."""
    backend = _make_conf()
    backend.set("view", "max_shapes_to_draw", 777, save=False)
    monkeypatch.setattr(confmod, "CONF", backend)
    options = DataLabOptions()
    from datalab.config.config_persistence import (  # pylint: disable=import-outside-toplevel
        load_options_from_ini,
    )

    load_options_from_ini(options, backend)

    assert options.max_shapes_to_draw.get(200) == 777
    assert backend.get("view", "max_shapes_to_draw") == 777


def test_field_get_none_does_not_create_ini_key(monkeypatch) -> None:
    """A None default returns the field value without creating an INI key."""
    backend = _make_conf()
    monkeypatch.setattr(confmod, "CONF", backend)
    options = DataLabOptions()
    options.set_ini_persist_enabled(True)

    assert options.plugins_enabled_list.get(None) is None
    assert not backend.has_option("main", "plugins_enabled_list")


def test_field_set_updates_ini_and_json(monkeypatch) -> None:
    """A field set updates the typed value, JSON environment, and INI backend."""
    backend = _make_conf()
    monkeypatch.setattr(confmod, "CONF", backend)
    options = DataLabOptions()
    options.set_ini_persist_enabled(True)

    options.available_memory_threshold.set(640)

    assert options.available_memory_threshold.get() == 640
    assert backend.get("main", "available_memory_threshold") == 640
    assert json.loads(os.environ[options.ENV_VAR])["available_memory_threshold"] == 640


def test_datalab_conf_is_active_sigimax_conf() -> None:
    """DataLab and reused SigimaX components share the same typed singleton."""
    from datalab.config import Conf  # pylint: disable=import-outside-toplevel

    assert get_conf() is Conf


def test_inherited_and_specific_options() -> None:
    """Inherited SigimaX options and DataLab-specific options coexist."""
    opt = DataLabOptions()

    # Inherited from SigimaXOptions.
    assert opt.color_mode.get() == "auto"
    assert opt.fft_shift_enabled.get() is True

    # DataLab-specific.
    assert opt.process_isolation_enabled.get() is True
    assert opt.rpc_server_enabled.get() is True
    assert opt.rpc_server_port.get() is None
    assert opt.plugins_enabled.get() is True

    # Former [macro]/[ai] sections are flattened with a prefix.
    assert opt.macro_console_max_lines.get() == 5000
    assert opt.ai_provider.get() == "openai"
    assert opt.ai_enabled.get() is False


def test_field_type_replacements() -> None:
    """Inherited path/dir fields are replaced by DataLab field types."""
    opt = DataLabOptions()

    assert isinstance(opt.traceback_log_path, ConfigPathOptionField)
    assert isinstance(opt.faulthandler_log_path, ConfigPathOptionField)
    assert isinstance(opt.base_dir, WorkingDirOptionField)

    # ConfigPathOptionField resolves the basename to an absolute path.
    resolved = opt.traceback_log_path.get()
    assert resolved.endswith(".DataLab_traceback.log")


def test_to_dict_is_json_serializable() -> None:
    """to_dict returns a JSON-serializable mapping, even with DataSet options."""
    opt = DataLabOptions()

    # Assign a DataSet option to ensure it is serialized as JSON, not an object.
    param = _SampleParam()
    param.value = 5
    opt.sig_shape_param.set(param)

    data = opt.to_dict()
    # Must be JSON-serializable (no DataSet instances or resolved paths).
    json.dumps(data)

    assert isinstance(opt.sig_shape_param, DataSetOptionField)
    assert data["sig_shape_param"] is not None
    # Unset DataSet options serialize to None.
    assert data["ima_shape_param"] is None
    # Config-path fields serialize to their raw basename, not the resolved path.
    assert data["traceback_log_path"] == ".DataLab_traceback.log"


def test_from_dict_round_trip() -> None:
    """from_dict restores values, including raw and DataSet fields."""
    opt = DataLabOptions()

    param = _SampleParam()
    param.value = 42
    opt.sig_shape_param.set(param)
    opt.process_isolation_enabled.set(False)
    opt.ai_provider.set("local")

    snapshot = opt.to_dict()

    # Mutate, then restore from the snapshot.
    opt.process_isolation_enabled.set(True)
    opt.ai_provider.set("openai")
    opt.sig_shape_param.set_raw(None)

    opt.from_dict(snapshot)

    assert opt.process_isolation_enabled.get() is False
    assert opt.ai_provider.get() == "local"
    assert opt.sig_shape_param.get().value == 42


def test_reset_to_defaults() -> None:
    """reset_to_defaults restores the captured default values."""
    opt = DataLabOptions()

    opt.process_isolation_enabled.set(False)
    opt.macro_console_max_lines.set(123)
    opt.base_dir.set_raw("/some/stale/dir")

    opt.reset_to_defaults()

    assert opt.process_isolation_enabled.get() is True
    assert opt.macro_console_max_lines.get() == 5000
    assert opt.base_dir.get_raw() == ""


def test_env_sync_is_valid_json() -> None:
    """Setting an option keeps the environment variable synchronized as JSON."""
    opt = DataLabOptions()
    opt.rpc_server_enabled.set(False)

    raw = opt.get_env()
    parsed = json.loads(raw)
    assert parsed["rpc_server_enabled"] is False
