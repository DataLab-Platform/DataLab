# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab application identity and resource paths
-------------------------------------------------

Application identity constants, configuration/legacy file path helpers,
translation and resource setup, and the plugin-path helpers that do not need
the shared :data:`~datalab.config.core.Conf` singleton.

This module has no dependency on the rest of :mod:`datalab.config`, so that
it can be imported first by :mod:`datalab.config.options`,
:mod:`datalab.config.persistence` and :mod:`datalab.config.core`.
"""

from __future__ import annotations

import logging
import os
import os.path as osp
import sys
from typing import TYPE_CHECKING

from guidata import configtools
from guidata.userconfig import get_config_basedir
from sigimax.config import is_frozen
from sigimax.utils.conf import Configuration

from datalab import __version__

if TYPE_CHECKING:
    from datalab.config.options import DataLabOptions

#: Application name used for configuration storage and default log basenames.
APP_NAME = "DataLab"
MOD_NAME = "datalab"
TYPED_CONFIG_SUFFIX = "_typed"


def get_config_app_name() -> str:
    """Get configuration application name with major version suffix.

    This function returns the application name used for configuration storage.
    Starting from v1.0, the major version is appended to allow different major
    versions to coexist on the same machine without interfering with each other.

    Returns:
        str: Configuration application name (e.g., "DataLab" for v0.x,
             "DataLab_v1" for v1.x)

    Examples:
        - v0.20.x → "DataLab" (configuration stored in ~/.DataLab)
        - v1.0.x → "DataLab_v1" (configuration stored in ~/.DataLab_v1)
        - v2.0.x → "DataLab_v2" (configuration stored in ~/.DataLab_v2)
    """
    major_version = __version__.split(".", maxsplit=1)[0]

    # Keep v0.x configuration folder unchanged for backward compatibility
    if major_version == "0":
        return APP_NAME

    return f"{APP_NAME}_v{major_version}"


def get_legacy_config_filename() -> str:
    """Return the configuration filename used by DataLab 1.2 and earlier."""
    config_name = get_config_app_name()
    return osp.join(get_config_basedir(), f".{config_name}", f"{config_name}.ini")


def get_typed_config_filename() -> str:
    """Return the typed configuration filename used by DataLab 1.3 and later."""
    config_name = get_config_app_name()
    return osp.join(
        get_config_basedir(),
        f".{config_name}",
        f"{config_name}{TYPED_CONFIG_SUFFIX}.ini",
    )


def get_config_path(basename: str) -> str:
    """Return a path inside the DataLab configuration directory."""
    return Configuration.get_path(basename)


def get_config_filename() -> str:
    """Return the DataLab INI configuration file name."""
    return Configuration.get_filename()


_ = configtools.get_translation(MOD_NAME)

APP_DESC = _("""DataLab is a generic signal and image processing platform""")

DEBUG = os.environ.get("DATALAB_DEBUG", "").lower() in ("1", "true")
if DEBUG:
    print("*** DEBUG mode *** [Reset configuration file, do not redirect std I/O]")

configtools.add_image_module_path(MOD_NAME, osp.join("data", "logo"))
configtools.add_image_module_path(MOD_NAME, osp.join("data", "icons"))

DATAPATH = configtools.get_module_data_path(MOD_NAME, "data")
SHOTPATH = osp.join(
    configtools.get_module_data_path(MOD_NAME), os.pardir, "doc", "images", "shots"
)
OTHER_PLUGINS_PATHLIST = [configtools.get_module_data_path(MOD_NAME, "plugins")]

IS_FROZEN = is_frozen(MOD_NAME)
if IS_FROZEN:
    OTHER_PLUGINS_PATHLIST.append(osp.join(osp.dirname(sys.executable), "plugins"))
    try:
        os.mkdir(OTHER_PLUGINS_PATHLIST[-1])
    except OSError:
        pass

PLUGIN_OK_COLOR = "#2ecc71"
PLUGIN_ERROR_COLOR = "#e74c3c"

# Additional third-party plugin directories provided via the `DATALAB_PLUGINS`
# environment variable. Multiple paths may be separated by `os.pathsep`
# (`;` on Windows, `:` on Unix), following the same convention as `PYTHONPATH`.
# Non-existent paths are skipped with a warning logged at startup.
DATALAB_PLUGINS_ENV_VAR = "DATALAB_PLUGINS"
#: Plugin directories declared through the ``DATALAB_PLUGINS`` env var
#: (subset of :data:`OTHER_PLUGINS_PATHLIST`, kept around so that consumers
#: such as the plugin configuration dialog can flag them as user-provided).
DATALAB_PLUGINS_ENV_PATHS: list[str] = []


def parse_datalab_plugins_env_var(
    env_value: str | None,
    pathlist: list[str],
    env_paths: list[str],
) -> None:
    """Parse ``DATALAB_PLUGINS`` and append valid directories to ``pathlist``.

    Args:
        env_value: Raw value of the ``DATALAB_PLUGINS`` environment variable
         (``None`` or empty string is a no-op).
        pathlist: Plugin search path list to extend in-place
         (typically :data:`OTHER_PLUGINS_PATHLIST`).
        env_paths: List of env-var-provided directories to extend in-place
         (typically :data:`DATALAB_PLUGINS_ENV_PATHS`), used by the GUI to
         flag entries originating from the environment variable.
    """
    if not env_value:
        return

    logger = logging.getLogger(__name__)
    for raw_path in env_value.split(os.pathsep):
        path = raw_path.strip()
        if not path:
            continue
        path = osp.normpath(osp.expanduser(path))
        if osp.isdir(path):
            if path not in pathlist:
                pathlist.append(path)
            if path not in env_paths:
                env_paths.append(path)
        else:
            logger.warning(
                "%s: ignoring non-existent plugin directory '%s'",
                DATALAB_PLUGINS_ENV_VAR,
                path,
            )


parse_datalab_plugins_env_var(
    os.environ.get(DATALAB_PLUGINS_ENV_VAR),
    OTHER_PLUGINS_PATHLIST,
    DATALAB_PLUGINS_ENV_PATHS,
)


def normalize_plugin_paths(paths: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize a list of plugin directories and drop duplicates/empties."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_path in paths or []:
        if not raw_path:
            continue
        path = osp.normpath(osp.abspath(osp.expanduser(raw_path)))
        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return normalized


def migrate_legacy_plugin_paths(
    options: DataLabOptions,
) -> list[str]:
    """Migrate the deprecated single plugin path into the typed path list."""
    candidates = list(options.plugins_path_list.get([]) or [])
    legacy_path = options.plugins_path.get("")
    fixed_default = osp.normpath(get_config_path("plugins"))
    if legacy_path and isinstance(legacy_path, str):
        normalized_legacy = osp.normpath(osp.abspath(osp.expanduser(legacy_path)))
        if not candidates and normalized_legacy != fixed_default:
            candidates.append(legacy_path)
            options.plugins_path_list.set(candidates)
    return normalize_plugin_paths(candidates)
