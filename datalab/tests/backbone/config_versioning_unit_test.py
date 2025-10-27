# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Configuration versioning test.

Test that the configuration folder name is versioned correctly.
"""

import datalab
from datalab.config import get_config_app_name


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


if __name__ == "__main__":
    test_config_app_name()
