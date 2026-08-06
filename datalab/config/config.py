# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration module
----------------------------

This module handles `DataLab` configuration (options, images and icons).
"""

from __future__ import annotations

import configparser
import logging
import os
import os.path as osp
import sys
import tempfile

from guidata import configtools
from guidata.dataset import BoolItem, StringItem
from guidata.userconfig import get_config_basedir
from plotpy.config import CONF as PLOTPY_CONF
from plotpy.styles import MarkerParam, ShapeParam
from sigima.proc.title_formatting import (
    PlaceholderTitleFormatter,
    set_default_title_formatter,
)
from sigimax.config import is_frozen, set_conf
from sigimax.utils import conf
from sigimax.utils.conf import AppUserConfig, Configuration

from datalab import __version__
from datalab.config.config_options import DataLabOptions
from datalab.config.config_persistence import (
    IniOptionStore,
    load_options_from_ini,
    save_options_to_ini,
)

# Configure Sigima to use DataLab-compatible placeholder title formatting
set_default_title_formatter(PlaceholderTitleFormatter())

CONF_VERSION = DataLabOptions.CONF_VERSION

APP_NAME = DataLabOptions.APP_NAME
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


class DataLabUserConfig(AppUserConfig):
    """DataLab INI backend keeping typed and legacy files side by side."""

    def filename(self) -> str:
        """Return the typed configuration filename."""
        return self.get_path(f"{self.name}{TYPED_CONFIG_SUFFIX}.ini")


class _LegacyConfigReader(AppUserConfig):
    """In-memory reader preventing writes to a legacy DataLab INI file."""

    def __init__(self, name: str) -> None:
        super().__init__({})
        self.name = name

    def set(self, section, option, value, verbose=False, save=True) -> None:
        """Update the snapshot in memory without writing the legacy file."""
        del save
        super().set(section, option, value, verbose=verbose, save=False)

    def save(self) -> None:
        """Do not persist the in-memory legacy snapshot."""

    def cleanup(self) -> None:
        """Do not delete the source legacy configuration."""

    def remove_option(self, section, option) -> bool:
        """Remove an option from memory without saving."""
        return configparser.ConfigParser.remove_option(self, section, option)

    def remove_section(self, section) -> bool:
        """Remove a section from memory without saving."""
        return configparser.ConfigParser.remove_section(self, section)


# Install the DataLab backend before consumers import ``CONF`` directly from
# ``sigimax.utils.conf``. No file is read or written until ``initialize()``.
if not isinstance(conf.CONF, DataLabUserConfig):
    conf.CONF = DataLabUserConfig({})


def atomic_save_configuration(config: AppUserConfig) -> None:
    """Atomically write a configuration backend to its target filename."""
    filename = config.filename()
    directory = osp.dirname(filename)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    descriptor, temporary_filename = tempfile.mkstemp(
        dir=directory,
        prefix=f".{osp.basename(filename)}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            config.write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_filename, filename)
    finally:
        if osp.exists(temporary_filename):
            os.remove(temporary_filename)


def migrate_legacy_configuration(
    options: DataLabOptions,
    legacy_filename: str,
    typed_conf: AppUserConfig,
) -> bool:
    """Initialize a missing typed configuration from a legacy INI file.

    Args:
        options: Typed DataLab options to initialize.
        legacy_filename: DataLab 1.2 configuration filename.
        typed_conf: DataLab 1.3 INI backend.

    Returns:
        True if the legacy configuration was migrated, False if migration was
         unnecessary or impossible because a source file was absent.
    """
    if osp.isfile(typed_conf.filename()) or not osp.isfile(legacy_filename):
        return False

    legacy_conf = _LegacyConfigReader(get_config_app_name())
    legacy_conf.read(legacy_filename, encoding="utf-8")
    load_options_from_ini(options, legacy_conf)
    migrate_legacy_plugin_paths(options)
    typed_conf.set_version(CONF_VERSION, save=False)
    save_options_to_ini(options, typed_conf, save=False)
    atomic_save_configuration(typed_conf)
    return True


_ = configtools.get_translation(MOD_NAME)

APP_DESC = _("""DataLab is a generic signal and image processing platform""")

DEBUG = os.environ.get("DATALAB_DEBUG", "").lower() in ("1", "true")
if DEBUG:
    print("*** DEBUG mode *** [Reset configuration file, do not redirect std I/O]")

TEST_SEGFAULT_ERROR = len(os.environ.get("TEST_SEGFAULT_ERROR", "")) > 0
if TEST_SEGFAULT_ERROR:
    print('*** TEST_SEGFAULT_ERROR mode *** [Enabling test action in "?" menu]')


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


def get_config_path(basename: str) -> str:
    """Return a path inside the DataLab configuration directory."""
    return Configuration.get_path(basename)


def get_config_filename() -> str:
    """Return the DataLab INI configuration file name."""
    return Configuration.get_filename()


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


PLUGIN_OK_COLOR = "#2ecc71"
PLUGIN_ERROR_COLOR = "#e74c3c"


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
