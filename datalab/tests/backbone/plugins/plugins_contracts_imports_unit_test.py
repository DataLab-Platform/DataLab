# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit test keeping the plugin contracts importable without the Qt plugin host."""

from __future__ import annotations

import subprocess
import sys

CONTRACT_MODULES = (
    "examples",
    "instruments",
    "recipe_binding",
    "recipes",
    "resources",
    "tiles",
    "tools",
)
# Installer modules, reused outside the Qt host (DataLab-Web, catalog tooling)
HEADLESS_HOST_MODULES = ("catalog", "store", "wheels")


def test_plugin_contracts_do_not_import_plugin_host() -> None:
    """Headless plugin layers import the contracts without loading Qt."""
    imports = "".join(
        f"import datalab.plugins.{name}\n"
        for name in CONTRACT_MODULES + HEADLESS_HOST_MODULES
    )
    code = (
        f"import sys\n{imports}"
        "print([name for name in ('datalab.plugins.base', 'qtpy') "
        "if name in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


if __name__ == "__main__":
    test_plugin_contracts_do_not_import_plugin_host()
