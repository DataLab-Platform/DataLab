# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration lifecycle and shared singleton
------------------------------------------------------

Initialization/reset lifecycle, the shared :data:`Conf` options singleton,
and PlotPy default instances tied to it (``DataLabShapeParam``).
"""

from __future__ import annotations

import os.path as osp

from guidata.dataset import BoolItem, StringItem
from plotpy.config import CONF as PLOTPY_CONF
from plotpy.styles import MarkerParam, ShapeParam
from sigima.proc.title_formatting import (
    PlaceholderTitleFormatter,
    set_default_title_formatter,
)
from sigimax.config import set_conf
from sigimax.utils import conf
from sigimax.utils.conf import Configuration

from datalab.config.appinfo import (
    DEBUG,
    get_config_app_name,
    get_config_path,
    get_legacy_config_filename,
    get_typed_config_filename,
    normalize_plugin_paths,
)
from datalab.config.options import DataLabOptions
from datalab.config.persistence import (
    CONF_VERSION,
    DataLabUserConfig,
    IniOptionStore,
    load_options_from_ini,
    migrate_legacy_configuration,
)

# Configure Sigima to use DataLab-compatible placeholder title formatting
set_default_title_formatter(PlaceholderTitleFormatter())


def get_user_plugin_paths() -> list[str]:
    """Return user-configured extra plugin directories.

    The deprecated single ``plugins_path`` value is converted once when the
    legacy configuration is migrated. Runtime reads only use the typed list so
    an explicitly empty list remains empty.
    """
    return normalize_plugin_paths(Conf.plugins_path_list.get([]) or [])


def set_user_plugin_paths(paths: list[str] | tuple[str, ...]) -> None:
    """Persist user-configured extra plugin directories.

    Writes to ``plugins_path_list``.  The deprecated ``plugins_path`` is left
    untouched so that older DataLab versions can still find at least one
    user-configured directory.
    """
    normalized = normalize_plugin_paths(list(paths))
    Conf.plugins_path_list.set(normalized)


#: Active initialization mode: ``None`` (not yet initialized), ``"user"``
#: (user configuration loaded/migrated, INI persistence enabled) or
#: ``"defaults"`` (production defaults only, no persistence).
_MODE: str | None = None


def _apply_runtime_defaults() -> None:
    """Apply defaults that depend on initialized application paths."""
    if not Conf.macro_templates_path.get():
        Conf.macro_templates_path.set(get_config_path("macro_templates"))
    Conf.sync_with_sigima()
    assert Conf.plot_toolbar_position.get() in ("top", "bottom", "left", "right")


def initialize(load_user_config: bool) -> None:
    """Initialize the shared DataLab options.

    Args:
        load_user_config: If True, load or migrate the user configuration and
         enable INI persistence. If False, use production defaults in memory.
    """
    global _MODE  # pylint: disable=global-statement
    requested_mode = "user" if load_user_config else "defaults"
    if _MODE is not None:
        if _MODE != requested_mode:
            raise RuntimeError("DataLab configuration is already initialized")
        return

    config_app_name = get_config_app_name()
    typed_exists = load_user_config and osp.isfile(get_typed_config_filename())
    Conf.detach_store()
    try:
        Conf.reset_to_defaults()
        if not isinstance(conf.CONF, DataLabUserConfig):
            conf.CONF = DataLabUserConfig({})
        Configuration.initialize(
            config_app_name, CONF_VERSION, load=typed_exists and not DEBUG
        )
        if load_user_config and not DEBUG:
            if typed_exists:
                load_options_from_ini(Conf, conf.CONF)
            elif not migrate_legacy_configuration(
                Conf, get_legacy_config_filename(), conf.CONF
            ):
                conf.CONF.set_version(CONF_VERSION, save=False)
        _apply_runtime_defaults()
        Conf.set_plotpy_application(config_app_name)
        _MODE = requested_mode
    finally:
        if _MODE == "user":
            Conf.attach_store(IniOptionStore(Conf))
        else:
            Conf.detach_store()


def ensure_initialized(load_user_config: bool) -> None:
    """Initialize configuration unless a caller already selected a mode."""
    if _MODE is None:
        initialize(load_user_config)


def reset_to_defaults() -> None:
    """Reset the shared options to production defaults without loading an INI."""
    Conf.detach_store()
    Conf.reset_to_defaults()
    _apply_runtime_defaults()


def reset():
    """Reset application configuration in the active initialization mode."""
    global _MODE  # pylint: disable=global-statement
    if _MODE == "defaults":
        reset_to_defaults()
        return

    Conf.detach_store()
    Configuration.reset()
    _MODE = None
    initialize(load_user_config=True)


class DataLabShapeParam(ShapeParam):
    """ShapeParam subclass with internal items hidden from settings dialog"""

    # Items are redeclared rather than hidden in place: guidata shares its
    # DataItem instances with subclasses, so ``set_prop`` would alter every
    # ``ShapeParam`` dialog in the application.
    label = StringItem("Title", default="").set_prop("display", hide=True)
    readonly = BoolItem("Read-only shape", default=False).set_prop("display", hide=True)
    private = BoolItem("Private shape", default=False).set_prop("display", hide=True)


# Configurations written by DataLab <= 1.2 refer to this class as
# ``datalab.config.DataLabShapeParam``, when ``datalab.config`` was a module.
DataLabShapeParam.__module__ = "datalab.config"


#: Active typed DataLab configuration shared with reused SigimaX components.
Conf: DataLabOptions = DataLabOptions()  # pylint: disable=invalid-name
set_conf(Conf)


def initialize_default_plotpy_instances():
    """Initialize default PlotPy instances for DataLab configuration options"""
    # PlotPy defaults have been applied by ``DataLabOptions.__init__`` (SigimaX)
    _sig_shapeparam = DataLabShapeParam()
    _sig_shapeparam.read_config(PLOTPY_CONF, "results", "s/annotation")
    Conf.sig_shape_param.set_default_instance(_sig_shapeparam)
    Conf.sig_shape_param.get()

    _sig_markerparam = MarkerParam()
    _sig_markerparam.read_config(PLOTPY_CONF, "results", "s/marker/cursor")
    Conf.sig_marker_param.set_default_instance(_sig_markerparam)
    Conf.sig_marker_param.get()

    _ima_shapeparam = DataLabShapeParam()
    _ima_shapeparam.read_config(PLOTPY_CONF, "results", "i/annotation")
    Conf.ima_shape_param.set_default_instance(_ima_shapeparam)
    Conf.ima_shape_param.get()

    _ima_markerparam = MarkerParam()
    _ima_markerparam.read_config(PLOTPY_CONF, "results", "i/marker/cursor")
    Conf.ima_marker_param.set_default_instance(_ima_markerparam)
    Conf.ima_marker_param.get()


initialize_default_plotpy_instances()
