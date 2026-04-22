# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the ``DATALAB_PLUGINS`` environment variable.

These tests exercise the parsing logic that turns the environment variable
into entries of :data:`datalab.config.OTHER_PLUGINS_PATHLIST` and verifies
that :func:`datalab.plugins.discover_plugins` actually finds plugin modules
placed in those directories.
"""

# guitest: skip

from __future__ import annotations

import importlib
import os
import sys
import textwrap
from pathlib import Path

import pytest


def _reload_config(monkeypatch: pytest.MonkeyPatch, env_value: str | None) -> object:
    """Reload ``datalab.config`` with a controlled ``DATALAB_PLUGINS`` value."""
    if env_value is None:
        monkeypatch.delenv("DATALAB_PLUGINS", raising=False)
    else:
        monkeypatch.setenv("DATALAB_PLUGINS", env_value)
    import datalab.config as config_mod

    return importlib.reload(config_mod)


def test_env_var_unset_leaves_default_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the env var is unset, the default plugin path list is unchanged."""
    config_mod = _reload_config(monkeypatch, None)
    assert config_mod.OTHER_PLUGINS_PATHLIST  # at least the bundled directory


def test_env_var_adds_existing_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Existing directories listed in the env var are appended to the path list."""
    dir1 = tmp_path / "plugins_a"
    dir2 = tmp_path / "plugins_b"
    dir1.mkdir()
    dir2.mkdir()
    env_value = os.pathsep.join([str(dir1), str(dir2)])

    config_mod = _reload_config(monkeypatch, env_value)
    paths = [os.path.normpath(p) for p in config_mod.OTHER_PLUGINS_PATHLIST]
    assert os.path.normpath(str(dir1)) in paths
    assert os.path.normpath(str(dir2)) in paths


def test_env_var_skips_missing_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing directories are skipped silently and a warning is logged."""
    existing = tmp_path / "exists"
    existing.mkdir()
    missing = tmp_path / "does_not_exist"
    env_value = os.pathsep.join([str(existing), str(missing), "  "])

    with caplog.at_level("WARNING", logger="datalab.config"):
        config_mod = _reload_config(monkeypatch, env_value)

    paths = [os.path.normpath(p) for p in config_mod.OTHER_PLUGINS_PATHLIST]
    assert os.path.normpath(str(existing)) in paths
    assert os.path.normpath(str(missing)) not in paths
    assert any("does_not_exist" in r.message for r in caplog.records)


def test_discover_plugins_finds_module_in_env_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``datalab_*`` module in a ``DATALAB_PLUGINS`` directory is discovered."""
    plugin_dir = tmp_path / "extra_plugins"
    plugin_dir.mkdir()
    plugin_name = "datalab_envvar_discovery_probe"
    (plugin_dir / f"{plugin_name}.py").write_text(
        textwrap.dedent(
            """
            \"\"\"Stub plugin used by DATALAB_PLUGINS unit test.\"\"\"
            DISCOVERED = True
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("DATALAB_PLUGINS", str(plugin_dir))
    # Ensure config picks up the env var, then reload plugins module so it
    # imports the refreshed OTHER_PLUGINS_PATHLIST.
    importlib.reload(importlib.import_module("datalab.config"))
    plugins_mod = importlib.reload(importlib.import_module("datalab.plugins"))

    # Make sure plugins are enabled for discovery
    from datalab.config import Conf

    monkeypatch.setattr(Conf.main.plugins_enabled, "get", lambda *a, **kw: True)

    try:
        modules = plugins_mod.discover_plugins()
        discovered_names = {m.__name__ for m in modules}
        assert plugin_name in discovered_names
    finally:
        sys.modules.pop(plugin_name, None)


if __name__ == "__main__":
    pytest.main([__file__])
