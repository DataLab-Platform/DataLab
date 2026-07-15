"""DataLab configuration package.

This package keeps backward-compatible package-level symbols for call sites
that still rely on ``datalab.config`` attributes while the implementation now
resides in submodules.
"""

from __future__ import annotations

from . import config
from .config import (
    APP_NAME,
    CONF_VERSION,
    DATAPATH,
    DEBUG,
    IS_FROZEN,
    MOD_NAME,
    PLOTPY_CONF,
    SHOTPATH,
    Conf,
    _,
    get_config_app_name,
    get_mod_source_dir,
    get_old_log_fname,
    initialize,
    reset,
)

__all__ = [
    "APP_NAME",
    "CONF_VERSION",
    "Conf",
    "DATAPATH",
    "DEBUG",
    "IS_FROZEN",
    "MOD_NAME",
    "PLOTPY_CONF",
    "SHOTPATH",
    "_",
    "config",
    "get_config_app_name",
    "get_mod_source_dir",
    "get_old_log_fname",
    "initialize",
    "reset",
]


def __getattr__(name: str):
    """Delegate unknown package attributes to ``datalab.config.config``.

    This preserves backward compatibility for legacy imports such as
    ``from datalab.config import PLOTPY_CONF`` while keeping implementation
    split across submodules.
    """
    return getattr(config, name)
