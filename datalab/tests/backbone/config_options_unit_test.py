# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the DataLab options container
(:mod:`datalab.config_options`).
"""

import json

import guidata.dataset as gds

from datalab.config.config_options import DataLabOptions
from datalab.config.optionfields import (
    ConfigPathOptionField,
    DataSetOptionField,
    WorkingDirOptionField,
)


class _SampleParam(gds.DataSet):
    """Simple DataSet used to exercise DataSet options."""

    value = gds.IntItem("Value", default=1)


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
