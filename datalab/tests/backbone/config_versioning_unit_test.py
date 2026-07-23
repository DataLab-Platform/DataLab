# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Configuration versioning test.

Test that the configuration folder name is versioned correctly.
"""

import os.path as osp

import datalab
from datalab.config import (
    DataLabUserConfig,
    create_config_backend,
    get_config_app_name,
    get_legacy_config_filename,
    get_typed_config_filename,
)
from datalab.config.config import LegacyConfigSnapshot


def test_config_app_name():
    """Test configuration app name versioning."""
    config_name = get_config_app_name()
    major_version = datalab.__version__.split(".", maxsplit=1)[0]

    print(f"DataLab version: {datalab.__version__}")
    print(f"Major version: {major_version}")
    print(f"Config app name: {config_name}")

    # For v0.x, the config name should be "DataLab" (no suffix)
    if major_version == "0":
        assert config_name == "DataLab", f"Expected 'DataLab', got '{config_name}'"
        print("✓ v0.x uses legacy config folder name: .DataLab")
    else:
        # For v1.x, v2.x, etc., the config name should include version suffix
        expected = f"DataLab_v{major_version}"
        assert config_name == expected, f"Expected '{expected}', got '{config_name}'"
        print(f"✓ v{major_version}.x uses versioned config folder: .{config_name}")


def test_v1_legacy_and_typed_config_files_share_directory() -> None:
    """DataLab 1.2 and 1.3 configs coexist in the v1 profile directory."""
    legacy_filename = get_legacy_config_filename()
    typed_filename = get_typed_config_filename()

    assert osp.basename(legacy_filename) == "DataLab_v1.ini"
    assert osp.basename(typed_filename) == "DataLab_v1_typed.ini"
    assert osp.dirname(legacy_filename) == osp.dirname(typed_filename)
    assert legacy_filename != typed_filename


def test_configuration_backend_can_be_selected_explicitly() -> None:
    """Development and tests may switch between legacy and typed files."""
    legacy_backend = create_config_backend("legacy")
    typed_backend = create_config_backend("typed")
    legacy_backend.name = "DataLab_v1"
    typed_backend.name = "DataLab_v1"

    assert isinstance(legacy_backend, LegacyConfigSnapshot)
    assert isinstance(typed_backend, DataLabUserConfig)
    assert osp.basename(legacy_backend.filename()) == "DataLab_v1.ini"
    assert osp.basename(typed_backend.filename()) == "DataLab_v1_typed.ini"


if __name__ == "__main__":
    test_config_app_name()
