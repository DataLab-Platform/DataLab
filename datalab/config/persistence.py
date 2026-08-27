# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration persistence (INI <-> options container)
-------------------------------------------------------------

Bridges the flat, SigimaX-style :class:`datalab.config.options.DataLabOptions`
container with DataLab's historical INI backend
(:data:`sigimax.utils.conf.CONF`, a guidata ``UserConfig``).

The INI file remains the on-disk format for backward compatibility with existing
user configurations. **Only categorized options are persisted**: the INI section
is the option's category (``field.category``, defined in :mod:`sigimax.config`
and extended by :mod:`datalab.config.options`), and an option left uncategorized
is simply ignored here.

The INI key is the option name, unless the field declares an explicit
``storage_key``, or its name carries its own category as a prefix
(``ai_``/``macro_``, historically section-local keys).

Values are (de)serialized through the SigimaX storage protocol
(:meth:`sigimax.config.OptionField.to_storage` /
:meth:`~sigimax.config.OptionField.from_storage`), so this module never
introspects field types:

- ``field.storage_suffixes`` lists the INI keys occupied by the field: empty
  means a single ``<key>``, otherwise one ``<key>_<suffix>`` per suffix (e.g.
  ``family`` / ``size`` / ``bold`` for fonts).
- ``field.storage_escape`` requests ``%`` -> ``%%`` escaping at the INI boundary
  (ConfigParser interpolation), the in-memory value being kept clean.
- ``to_storage()`` returning ``None`` means "nothing to persist": the mapped INI
  keys are removed.

Type coercion is delegated to ``UserConfig`` (see
:meth:`guidata.userconfig.UserConfig.get`) by registering each field's raw
default with :meth:`~guidata.userconfig.UserConfig.set_default` before reading.
Unlike passing ``default=`` to ``get``, this registers the type without writing
the default back to the INI file.
"""

from __future__ import annotations

import configparser
import os
import os.path as osp
import tempfile
from typing import TYPE_CHECKING, Protocol

from sigimax.utils import conf as _confmod
from sigimax.utils.conf import AppUserConfig

from datalab.config.appinfo import (
    TYPED_CONFIG_SUFFIX,
    get_config_app_name,
    migrate_legacy_plugin_paths,
)
from datalab.config.options import DataLabOptions

if TYPE_CHECKING:
    from guidata.userconfig import UserConfig
    from sigimax.config import OptionField

CONF_VERSION = DataLabOptions.CONF_VERSION


class OptionStore(Protocol):
    """Persistence backend consulted by :class:`DataLabOptions` option hooks.

    This narrow protocol is what :mod:`datalab.config.options` depends on,
    instead of this module's free functions or the ``UserConfig`` backend
    directly, so that the two modules do not import each other.
    """

    def load_all(self) -> None:
        """Load all mapped option values from the backend."""

    def save_all(self, save: bool = True) -> None:
        """Save all mapped option values to the backend."""

    def save(self, name: str) -> None:
        """Save a single option value to the backend."""

    def has(self, name: str) -> bool:
        """Return whether an option has a value in the backend."""

    def remove(self, name: str) -> bool:
        """Remove an option from the backend."""


def _default_conf() -> UserConfig:
    """Return the live DataLab INI backend (resolved dynamically).

    Resolving the backend lazily (rather than importing it once at module load)
    is required because :meth:`sigimax.utils.conf.Configuration.reset` rebinds
    the module-level ``CONF`` singleton to a fresh instance.

    Returns:
        The current ``UserConfig`` INI backend.
    """
    return _confmod.CONF


#: Inherited SigimaX options that are set programmatically at startup (or are
#: purely presentation metadata), hence left uncategorized on purpose. Used by
#: the configuration completeness test as an allowlist.
EXPECTED_UNCATEGORIZED: frozenset[str] = frozenset(
    {
        "app_name",
        "app_version",
        "app_logo_path",
        "app_desc",
        "app_local_doc_path",
        "app_docurl",
        "app_homeurl",
        "app_supporturl",
        "app_developer",
        "app_copyright",
        "splash_image_path",
        "splash_show_progress",
        "datetime_format",
    }
)


def get_ini_location(options: DataLabOptions, name: str) -> tuple[str, str] | None:
    """Return the ``(section, ini_key)`` INI location of an option field.

    Args:
        options: The DataLab options container.
        name: The option field name.

    Returns:
        The ``(section, ini_key)`` pair, or ``None`` when the field is
         uncategorized (and therefore not persisted).
    """
    ini_key = options.get_field_ui_key(name)
    if ini_key is None:
        return None
    return options.get_field_category(name), ini_key


def _field_ini_keys(field: OptionField | None, ini_key: str) -> list[str]:
    """Return the INI keys occupied by an option field.

    Args:
        field: The option field (``None`` is treated as a single-key field).
        ini_key: The base INI key of the field.

    Returns:
        The list of INI keys: ``[ini_key]`` for a single-key field, one
         ``<ini_key>_<suffix>`` key per declared storage suffix otherwise.
    """
    suffixes = getattr(field, "storage_suffixes", ())
    if not suffixes:
        return [ini_key]
    return [f"{ini_key}_{suffix}" for suffix in suffixes]


def has_persisted_option(
    options: DataLabOptions, name: str, conf: UserConfig | None = None
) -> bool:
    """Return whether an option has a value in the INI backend.

    Args:
        options: The DataLab options container.
        name: The flat option field name.
        conf: The ``UserConfig`` INI backend to inspect (defaults to the
         module-level DataLab ``CONF``).

    Returns:
        True if the mapped INI option exists, False otherwise.
    """
    location = get_ini_location(options, name)
    if location is None:
        return False
    conf = _default_conf() if conf is None else conf
    section, ini_key = location
    field = getattr(options, name, None)
    return all(conf.has_option(section, key) for key in _field_ini_keys(field, ini_key))


def remove_persisted_option(
    options: DataLabOptions, name: str, conf: UserConfig | None = None
) -> bool:
    """Remove an option from the INI backend.

    Args:
        options: The DataLab options container.
        name: The flat option field name.
        conf: The ``UserConfig`` INI backend to update (defaults to the
         module-level DataLab ``CONF``).

    Returns:
        True if the mapped INI option existed and was removed, False otherwise.
    """
    location = get_ini_location(options, name)
    if location is None:
        return False
    conf = _default_conf() if conf is None else conf
    section, ini_key = location
    field = getattr(options, name, None)
    keys = [
        key for key in _field_ini_keys(field, ini_key) if conf.has_option(section, key)
    ]
    if not keys:
        return False
    for key in keys:
        conf.remove_option(section, key)
    conf.save()
    return True


def _iter_persisted_field_names(options: DataLabOptions):
    """Yield persisted (categorized) option field names in category order."""
    for _category, names in options.fields_by_category().items():
        yield from names


def _escape_percent(value: str) -> str:
    """Escape ``%`` as ``%%`` for ConfigParser interpolation."""
    return value.replace("%", "%%")


def _unescape_percent(value: str) -> str:
    """Unescape ``%%`` back to ``%``."""
    return value.replace("%%", "%")


def _load_field(options, conf, field_name: str, section: str, ini_key: str) -> None:
    """Load a single option field value from the INI backend.

    A stored value that the field cannot restore (invalid or obsolete) is
    removed from the INI file, so that a corrupted entry heals itself instead of
    breaking every subsequent startup.

    Args:
        options: The DataLab options container.
        conf: The ``UserConfig`` INI backend to read from.
        field_name: The flat option field name.
        section: The INI section name.
        ini_key: The INI option key.
    """
    field = getattr(options, field_name, None)
    if field is None:
        return
    keys = _field_ini_keys(field, ini_key)
    default_raw = options.get_default_raw(field_name)
    defaults = list(default_raw) if len(keys) > 1 else [default_raw]
    values = []
    found = False
    for key, key_default in zip(keys, defaults):
        if key_default is not None:
            # Register the expected type for ``conf.get`` coercion. Unlike
            # passing ``default=`` to ``conf.get``, this does not write the
            # default value back to the INI file.
            conf.set_default(section, key, key_default)
        if not conf.has_option(section, key):
            values.append(key_default)
            continue
        found = True
        value = conf.get(section, key)
        if field.storage_escape and isinstance(value, str):
            value = _unescape_percent(value)
        values.append(value)
    if not found:
        return
    try:
        field.from_storage(tuple(values) if len(keys) > 1 else values[0])
    except Exception:  # pylint: disable=broad-except
        remove_persisted_option(options, field_name, conf)
    else:
        if field.to_storage() is None:
            remove_persisted_option(options, field_name, conf)


def _save_field(options, conf, field_name: str, section: str, ini_key: str) -> None:
    """Save a single option field value to the INI backend (no file flush).

    A field whose storage value is ``None`` has nothing to persist: its INI keys
    are removed, so that a previously stored value does not linger.

    Args:
        options: The DataLab options container.
        conf: The ``UserConfig`` INI backend to write to.
        field_name: The flat option field name.
        section: The INI section name.
        ini_key: The INI option key.
    """
    field = getattr(options, field_name, None)
    if field is None:
        return
    keys = _field_ini_keys(field, ini_key)
    stored = field.to_storage()
    if stored is None:
        for key in keys:
            if conf.has_option(section, key):
                conf.remove_option(section, key)
        return
    values = list(stored) if len(keys) > 1 else [stored]
    for key, value in zip(keys, values):
        if field.storage_escape and isinstance(value, str):
            value = _escape_percent(value)
        conf.set(section, key, value, save=False)


def load_options_from_ini(
    options: DataLabOptions, conf: UserConfig | None = None
) -> None:
    """Load all mapped option values from the INI backend into the container.

    Args:
        options: The DataLab options container to populate.
        conf: The ``UserConfig`` INI backend to read from (defaults to the
         module-level DataLab ``CONF``).
    """
    conf = _default_conf() if conf is None else conf
    for field_name in _iter_persisted_field_names(options):
        location = get_ini_location(options, field_name)
        if location is None:
            continue
        _load_field(options, conf, field_name, *location)


def save_options_to_ini(
    options: DataLabOptions, conf: UserConfig | None = None, save: bool = True
) -> None:
    """Save all mapped option values from the container to the INI backend.

    Args:
        options: The DataLab options container to serialize.
        conf: The ``UserConfig`` INI backend to write to (defaults to the
         module-level DataLab ``CONF``).
        save: If True, flush the configuration file to disk once at the end.
    """
    conf = _default_conf() if conf is None else conf
    for field_name in _iter_persisted_field_names(options):
        if getattr(getattr(options, field_name, None), "runtime", False):
            # Runtime IPC value: persisted only by its owner, through
            # :func:`save_runtime_option`.
            continue
        location = get_ini_location(options, field_name)
        if location is None:
            continue
        _save_field(options, conf, field_name, *location)
    if save:
        conf.save()


def save_runtime_option(
    options: DataLabOptions, name: str, conf: UserConfig | None = None
) -> None:
    """Persist a single runtime option directly to the INI (single-key write).

    Fields declared ``runtime=True`` are excluded from the bulk
    :func:`save_options_to_ini` so that unrelated saves cannot clobber them.
    Their owner (e.g. the XML-RPC server writing its port) persists them through
    this authoritative single-key write.

    Args:
        options: The DataLab options container.
        name: The option field name to persist.
        conf: The ``UserConfig`` INI backend to write to (defaults to the
         module-level DataLab ``CONF``).
    """
    if conf is None and not options.is_ini_persist_enabled():
        return
    conf = _default_conf() if conf is None else conf
    location = get_ini_location(options, name)
    if location is None:
        return
    _save_field(options, conf, name, *location)
    conf.save()


def get_uncategorized_fields(options: DataLabOptions) -> list[str]:
    """Return option fields that are uncategorized and not explicitly excluded.

    Used by the configuration completeness test to guarantee that every option
    is either categorized (hence persisted) or intentionally excluded (in
    :data:`EXPECTED_UNCATEGORIZED`).

    Args:
        options: The DataLab options container to inspect.

    Returns:
        Sorted list of uncategorized option field names missing from
         :data:`EXPECTED_UNCATEGORIZED`.
    """
    from sigima.config import OptionField  # pylint: disable=import-outside-toplevel

    unexpected: list[str] = []
    for name in vars(options):
        if not isinstance(getattr(options, name), OptionField):
            continue
        if options.get_field_category(name):
            continue
        if name in EXPECTED_UNCATEGORIZED:
            continue
        unexpected.append(name)
    return sorted(unexpected)


class IniOptionStore:
    """:class:`OptionStore` backed by a DataLab INI ``UserConfig``.

    Args:
        options: The DataLab options container to persist.
        conf: The ``UserConfig`` INI backend to use (defaults to the live
         module-level DataLab ``CONF``, resolved dynamically on each call).
    """

    def __init__(self, options: DataLabOptions, conf: UserConfig | None = None) -> None:
        self._options = options
        self._conf = conf

    def load_all(self) -> None:
        """Load all mapped option values from the INI backend."""
        load_options_from_ini(self._options, self._conf)

    def save_all(self, save: bool = True) -> None:
        """Save all mapped option values to the INI backend."""
        save_options_to_ini(self._options, self._conf, save=save)

    def save(self, name: str) -> None:
        """Save a single option value to the INI backend."""
        conf = _default_conf() if self._conf is None else self._conf
        location = get_ini_location(self._options, name)
        if location is None:
            return
        _save_field(self._options, conf, name, *location)
        conf.save()

    def has(self, name: str) -> bool:
        """Return whether an option has a value in the INI backend."""
        return has_persisted_option(self._options, name, self._conf)

    def remove(self, name: str) -> bool:
        """Remove an option from the INI backend."""
        return remove_persisted_option(self._options, name, self._conf)


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
if not isinstance(_confmod.CONF, DataLabUserConfig):
    _confmod.CONF = DataLabUserConfig({})

# The configuration directory name only depends on the application version, so it is
# bound as soon as the backend is installed: ``get_config_path`` may be called at
# import time - e.g. by ``datalab.plugins`` - long before ``initialize()`` has run, and
# an unnamed guidata ``UserConfig`` would silently resolve to ``~/.none``.
_confmod.CONF.set_application(get_config_app_name(), CONF_VERSION, load=False)


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
