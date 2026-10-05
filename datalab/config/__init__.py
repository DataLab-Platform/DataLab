"""DataLab typed configuration package.

The public :data:`Conf` singleton is a flat
:class:`~datalab.config.options.DataLabOptions` container shared with
SigimaX. INI persistence and lifecycle helpers remain module-level functions.
"""

from __future__ import annotations

from . import core
from .appinfo import (
    APP_DESC,
    APP_NAME,
    DATALAB_PLUGINS_ENV_PATHS,
    DATALAB_PLUGINS_ENV_VAR,
    DATAPATH,
    DEBUG,
    IS_FROZEN,
    MOD_NAME,
    OTHER_PLUGINS_PATHLIST,
    PLUGIN_ERROR_COLOR,
    PLUGIN_OK_COLOR,
    SHOTPATH,
    _,
    get_config_filename,
    get_config_path,
    normalize_plugin_paths,
)
from .core import (
    PLOTPY_CONF,
    Conf,
    DataLabShapeParam,
    ensure_initialized,
    get_user_plugin_paths,
    initialize,
    reset,
    reset_to_defaults,
    set_user_plugin_paths,
)
from .persistence import save_runtime_option

__all__ = [
    "APP_DESC",
    "APP_NAME",
    "DATALAB_PLUGINS_ENV_PATHS",
    "DATALAB_PLUGINS_ENV_VAR",
    "DATAPATH",
    "DEBUG",
    "IS_FROZEN",
    "MOD_NAME",
    "OTHER_PLUGINS_PATHLIST",
    "PLOTPY_CONF",
    "PLUGIN_ERROR_COLOR",
    "PLUGIN_OK_COLOR",
    "SHOTPATH",
    "Conf",
    "DataLabShapeParam",
    "_",
    "core",
    "ensure_initialized",
    "get_config_filename",
    "get_config_path",
    "get_user_plugin_paths",
    "initialize",
    "normalize_plugin_paths",
    "reset",
    "reset_to_defaults",
    "save_runtime_option",
    "set_user_plugin_paths",
]
