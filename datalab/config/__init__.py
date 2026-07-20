"""DataLab configuration package.

This package keeps backward-compatible package-level symbols for call sites
that still rely on ``datalab.config`` attributes while the implementation now
resides in submodules.
"""

from __future__ import annotations

from . import config
from .config import (
    APP_DESC,
    APP_NAME,
    CONF_VERSION,
    DATALAB_PLUGINS_ENV_PATHS,
    DATALAB_PLUGINS_ENV_VAR,
    DATAPATH,
    DEBUG,
    IS_FROZEN,
    MOD_NAME,
    OTHER_PLUGINS_PATHLIST,
    PLOTPY_CONF,
    PLOTPY_DEFAULTS,
    PLUGIN_ERROR_COLOR,
    PLUGIN_OK_COLOR,
    SHOTPATH,
    TEST_SEGFAULT_ERROR,
    Conf,
    _,
    get_config_app_name,
    get_config_filename,
    get_config_path,
    get_mod_source_dir,
    get_old_log_fname,
    get_user_plugin_paths,
    initialize,
    normalize_plugin_paths,
    reload_from_ini,
    reset,
    set_user_plugin_paths,
)

__all__ = [
    "APP_DESC",
    "APP_NAME",
    "CONF_VERSION",
    "DATALAB_PLUGINS_ENV_PATHS",
    "DATALAB_PLUGINS_ENV_VAR",
    "DATAPATH",
    "DEBUG",
    "IS_FROZEN",
    "MOD_NAME",
    "OTHER_PLUGINS_PATHLIST",
    "PLOTPY_CONF",
    "PLOTPY_DEFAULTS",
    "PLUGIN_ERROR_COLOR",
    "PLUGIN_OK_COLOR",
    "SHOTPATH",
    "TEST_SEGFAULT_ERROR",
    "Conf",
    "_",
    "config",
    "get_config_app_name",
    "get_config_filename",
    "get_config_path",
    "get_mod_source_dir",
    "get_old_log_fname",
    "get_user_plugin_paths",
    "initialize",
    "normalize_plugin_paths",
    "reload_from_ini",
    "reset",
    "set_user_plugin_paths",
]


def __getattr__(name: str):
    """Delegate unknown package attributes to ``datalab.config.config``.

    This preserves backward compatibility for legacy imports such as
    ``from datalab.config import PLOTPY_CONF`` while keeping implementation
    split across submodules.
    """
    return getattr(config, name)
