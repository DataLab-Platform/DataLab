# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for transitional legacy configuration behavior."""

import json
import os

from sigimax.utils import conf as confmod
from sigimax.utils.conf import AppUserConfig

from datalab.config._legacy_bridge import _OptionProxy
from datalab.config.config_options import DataLabOptions


def _make_conf() -> AppUserConfig:
    """Return an isolated in-memory UserConfig backend for tests."""
    conf = AppUserConfig({})
    conf.set_application("DataLab_bridge_pytest", "1.0.0", load=False)
    return conf


def test_get_default_initializes_missing_ini_and_json(monkeypatch) -> None:
    """A non-None get default initializes both persistence representations."""
    backend = _make_conf()
    monkeypatch.setattr(confmod, "CONF", backend)
    options = DataLabOptions()
    options.set_ini_persist_enabled(True)
    proxy = _OptionProxy(options, options.available_memory_threshold)

    assert proxy.get(640) == 640
    assert backend.get("main", "available_memory_threshold") == 640
    assert json.loads(os.environ[options.ENV_VAR])["available_memory_threshold"] == 640


def test_get_default_returns_exact_supplied_value(monkeypatch) -> None:
    """Initialization returns the default itself, before field normalization."""
    backend = _make_conf()
    monkeypatch.setattr(confmod, "CONF", backend)
    options = DataLabOptions()
    options.set_ini_persist_enabled(True)
    proxy = _OptionProxy(options, options.window_size)
    default = [900, 700]

    assert proxy.get(default) is default
    assert options.window_size.get() == (900, 700)
    assert backend.get("main", "window_size") == (900, 700)


def test_get_default_preserves_existing_ini_value(monkeypatch) -> None:
    """An existing key is not replaced, even when equal to the field default."""
    backend = _make_conf()
    backend.set("view", "max_shapes_to_draw", 1000, save=False)
    monkeypatch.setattr(confmod, "CONF", backend)
    options = DataLabOptions()
    proxy = _OptionProxy(options, options.max_shapes_to_draw)

    assert proxy.get(200) == 1000
    assert backend.get("view", "max_shapes_to_draw") == 1000


def test_get_none_does_not_initialize_missing_ini(monkeypatch) -> None:
    """A None default returns the field value without creating an INI key."""
    backend = _make_conf()
    monkeypatch.setattr(confmod, "CONF", backend)
    options = DataLabOptions()
    proxy = _OptionProxy(options, options.plugins_enabled_list)

    assert proxy.get(None) is None
    assert not backend.has_option("main", "plugins_enabled_list")
