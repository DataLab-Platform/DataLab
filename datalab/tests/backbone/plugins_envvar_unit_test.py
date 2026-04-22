# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for the ``DATALAB_PLUGINS`` environment variable.

These tests exercise the parsing logic that turns the environment variable
into entries of :data:`datalab.config.OTHER_PLUGINS_PATHLIST` and verifies
that :func:`datalab.plugins.discover_plugins` actually finds plugin modules
placed in those directories.

The parsing function is tested directly (without reloading
``datalab.config``) to avoid corrupting global singletons (``Conf``,
``PluginRegistry``, the ``PluginBase`` metaclass) shared with the rest of
the test session.
"""

# guitest: skip

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from datalab import config as config_mod
from datalab import plugins as plugins_mod


def test_env_var_unset_leaves_default_paths() -> None:
    """When the env var is empty/unset, the path list is left untouched."""
    pathlist: list[str] = ["/initial/path"]
    env_paths: list[str] = []

    config_mod.parse_datalab_plugins_env_var(None, pathlist, env_paths)
    config_mod.parse_datalab_plugins_env_var("", pathlist, env_paths)

    assert pathlist == ["/initial/path"]
    assert env_paths == []


def test_env_var_adds_existing_directories(tmp_path: Path) -> None:
    """Existing directories listed in the env var are appended to both lists."""
    dir1 = tmp_path / "plugins_a"
    dir2 = tmp_path / "plugins_b"
    dir1.mkdir()
    dir2.mkdir()
    env_value = os.pathsep.join([str(dir1), str(dir2)])

    pathlist: list[str] = []
    env_paths: list[str] = []
    config_mod.parse_datalab_plugins_env_var(env_value, pathlist, env_paths)

    expected = [os.path.normpath(str(dir1)), os.path.normpath(str(dir2))]
    assert [os.path.normpath(p) for p in pathlist] == expected
    assert [os.path.normpath(p) for p in env_paths] == expected


def test_env_var_skips_missing_directories(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing directories are skipped and a warning is logged."""
    existing = tmp_path / "exists"
    existing.mkdir()
    missing = tmp_path / "does_not_exist"
    env_value = os.pathsep.join([str(existing), str(missing), "  "])

    pathlist: list[str] = []
    env_paths: list[str] = []
    with caplog.at_level("WARNING", logger="datalab.config"):
        config_mod.parse_datalab_plugins_env_var(env_value, pathlist, env_paths)

    paths = [os.path.normpath(p) for p in pathlist]
    assert os.path.normpath(str(existing)) in paths
    assert os.path.normpath(str(missing)) not in paths
    assert any("does_not_exist" in r.message for r in caplog.records)


def test_discover_plugins_finds_module_in_env_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ``datalab_*`` module in an env-var directory is discovered.

    ``OTHER_PLUGINS_PATHLIST`` is mutated in-place (and restored after) to
    avoid reloading ``datalab.config`` or ``datalab.plugins``, which would
    invalidate the ``PluginBase`` metaclass and break later tests.
    """
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

    extra_path = str(plugin_dir)
    config_mod.OTHER_PLUGINS_PATHLIST.append(extra_path)
    sys.modules.pop(plugin_name, None)
    try:
        # Ensure plugins are considered enabled regardless of user config
        monkeypatch.setattr(
            config_mod.Conf.main.plugins_enabled, "get", lambda *a, **kw: True
        )
        modules = plugins_mod.discover_plugins()
        discovered_names = {m.__name__ for m in modules}
        assert plugin_name in discovered_names
    finally:
        try:
            config_mod.OTHER_PLUGINS_PATHLIST.remove(extra_path)
        except ValueError:
            pass
        sys.modules.pop(plugin_name, None)


if __name__ == "__main__":
    pytest.main([__file__])
