"""DataLab typed configuration package.

The public :data:`Conf` singleton is a flat
:class:`~datalab.config.config_options.DataLabOptions` container shared with
SigimaX. INI persistence and lifecycle helpers remain module-level functions.
"""

from __future__ import annotations

from . import config
from .config import (
    APP_DESC,
    APP_NAME,
    CONF_VERSION,
    CONFIG_BACKEND_ENV_VAR,
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
    DataLabUserConfig,
    _,
    create_config_backend,
    get_config_app_name,
    get_config_backend_mode,
    get_config_filename,
    get_config_path,
    get_legacy_config_filename,
    get_mod_source_dir,
    get_old_log_fname,
    get_typed_config_filename,
    get_user_plugin_paths,
    initialize,
    migrate_legacy_configuration,
    migrate_legacy_plugin_paths,
    normalize_plugin_paths,
    reload_from_ini,
    reset,
    set_user_plugin_paths,
)

__all__ = [
    "APP_DESC",
    "APP_NAME",
    "CONFIG_BACKEND_ENV_VAR",
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
    "DataLabUserConfig",
    "_",
    "config",
    "create_config_backend",
    "get_config_app_name",
    "get_config_backend_mode",
    "get_config_filename",
    "get_config_path",
    "get_legacy_config_filename",
    "get_mod_source_dir",
    "get_old_log_fname",
    "get_typed_config_filename",
    "get_user_plugin_paths",
    "initialize",
    "migrate_legacy_configuration",
    "migrate_legacy_plugin_paths",
    "normalize_plugin_paths",
    "reload_from_ini",
    "reset",
    "set_user_plugin_paths",
]


def __getattr__(name: str):
    """Delegate implementation constants to ``datalab.config.config``."""
    return getattr(config, name)
