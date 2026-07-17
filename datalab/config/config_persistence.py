# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration persistence (INI <-> options container)
-------------------------------------------------------------

Bridges the flat, SigimaX-style :class:`datalab.config_options.DataLabOptions`
container with DataLab's historical INI backend
(:data:`sigimax.utils.conf.CONF`, a guidata ``UserConfig``).

The INI file remains the on-disk format for backward compatibility with existing
user configurations. The **INI section is derived from each option's category**
(``field.category``, defined in :mod:`sigimax.config` and extended by
:mod:`datalab.config_options`). The INI key defaults to the option name, with a
small set of exceptions (see :data:`INI_KEY_OVERRIDES` and the ``ai_``/``macro_``
prefix rule).

Encoding is derived from each field's *type*:

- :class:`~datalab.utils.optionfields.DataSetOptionField`: JSON string, with
  ``%`` escaped for ConfigParser.
- :class:`~datalab.utils.optionfields.FontOptionField`: three INI keys
  (``<key>_family`` / ``<key>_size`` / ``<key>_bold``).
- :class:`~datalab.utils.optionfields.ConfigPathOptionField` /
  :class:`~datalab.utils.optionfields.WorkingDirOptionField`: the raw stored
  value (basename / directory), via ``get_raw`` / ``set_raw``.
- datetime format fields (see :data:`DATETIME_FIELDS`): stored escaped
  (``%`` -> ``%%``); the in-memory value is kept clean.
- everything else: relies on ``UserConfig``'s ``repr``/``eval`` type coercion.

Type coercion is delegated to ``UserConfig`` (see
:meth:`guidata.userconfig.UserConfig.get`): passing each field's raw default as
the ``default`` argument registers the correct type and preserves the exact
behaviour of the legacy DataLab configuration system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sigimax.utils import conf as _confmod

from datalab.config.optionfields import (
    ConfigPathOptionField,
    DataSetOptionField,
    FontOptionField,
    WorkingDirOptionField,
)

if TYPE_CHECKING:
    from guidata.userconfig import UserConfig

    from datalab.config.config_options import DataLabOptions


def _default_conf() -> UserConfig:
    """Return the live DataLab INI backend (resolved dynamically).

    Resolving the backend lazily (rather than importing it once at module load)
    is required because :meth:`sigimax.utils.conf.Configuration.reset` rebinds
    the module-level ``CONF`` singleton to a fresh instance.

    Returns:
        The current ``UserConfig`` INI backend.
    """
    return _confmod.CONF


#: Fields storing ``strftime`` format strings. Their ``%`` characters must be
#: escaped as ``%%`` in the INI file (ConfigParser interpolation), while the
#: in-memory value is kept in clean form (``%H:%M:%S``).
DATETIME_FIELDS = frozenset({"sig_datetime_format_s", "sig_datetime_format_ms"})

#: Inherited SigimaX options that are set programmatically at startup (or are
#: purely presentation metadata) and are therefore not persisted to the INI file.
NON_PERSISTED: frozenset[str] = frozenset(
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

#: Runtime IPC fields that must NOT be written by the bulk container save. They
#: are shared across processes through the INI (e.g. the XML-RPC server port is
#: written by the running instance and read by remote clients). A bulk save
#: triggered by an unrelated option change (e.g. window geometry persisted on
#: close by another DataLab instance) would otherwise clobber the value written
#: by the current server. These fields stay categorized and are still *loaded*
#: from the INI; their owner persists them via :func:`save_runtime_option`.
RUNTIME_FIELDS: frozenset[str] = frozenset({"rpc_server_port"})

#: Categories whose fields drop their ``<category>_`` prefix when mapped to the
#: (historically section-local) INI key.
_PREFIX_SECTIONS = frozenset({"ai", "macro"})

#: Explicit INI key overrides for fields whose historical INI key differs from
#: the option name (and is not covered by the prefix rule).
INI_KEY_OVERRIDES: dict[str, str] = {
    "console_max_line_count": "max_line_count",
}


def get_ini_location(options: DataLabOptions, name: str) -> tuple[str, str] | None:
    """Return the ``(section, ini_key)`` INI location of an option field.

    The section is the field's category; the INI key is the option name, unless
    overridden by :data:`INI_KEY_OVERRIDES` or stripped of its ``ai_``/``macro_``
    prefix.

    Args:
        options: The DataLab options container.
        name: The option field name.

    Returns:
        The ``(section, ini_key)`` pair, or ``None`` when the field is
         uncategorized (and therefore not persisted).
    """
    section = options.get_field_category(name)
    if not section:
        return None
    if name in INI_KEY_OVERRIDES:
        ini_key = INI_KEY_OVERRIDES[name]
    elif section in _PREFIX_SECTIONS and name.startswith(f"{section}_"):
        ini_key = name[len(section) + 1 :]
    else:
        ini_key = name
    return section, ini_key


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
    default_raw = options.get_default_raw(field_name)

    if isinstance(field, DataSetOptionField):
        raw = conf.get(section, ini_key, default="")
        if isinstance(raw, str) and raw:
            try:
                field.from_json(_unescape_percent(raw))
            except Exception:  # pylint: disable=broad-except
                # Corrupted JSON: keep the default instance.
                pass
    elif isinstance(field, FontOptionField):
        fam_d, size_d, bold_d = default_raw
        family = conf.get(section, f"{ini_key}_family", default=fam_d)
        size = conf.get(section, f"{ini_key}_size", default=size_d)
        bold = conf.get(section, f"{ini_key}_bold", default=bold_d)
        field.set((family, size, bold), sync_env=False)
    elif isinstance(field, (ConfigPathOptionField, WorkingDirOptionField)):
        default = default_raw if default_raw is not None else ""
        field.set_raw(conf.get(section, ini_key, default=default))
    elif field_name in DATETIME_FIELDS:
        raw = conf.get(section, ini_key, default=_escape_percent(default_raw))
        field.set(_unescape_percent(raw), sync_env=False)
    else:
        # ``conf.get`` on a *missing* option re-persists the supplied default
        # through the INI backend, coercing it to the type inferred from any
        # previously stored value (e.g. ``int`` for ``rpc_server_port``).
        # Coercing a ``None`` default that way raises ``TypeError``
        # (``int(None)``). Read the stored value only when it exists and fall
        # back to the raw default otherwise (mirroring ``_save_field``, which
        # clears None-valued options instead of persisting them).
        if conf.has_option(section, ini_key):
            field.set(conf.get(section, ini_key), sync_env=False)
        else:
            field.set(default_raw, sync_env=False)


def _save_field(options, conf, field_name: str, section: str, ini_key: str) -> None:
    """Save a single option field value to the INI backend (no file flush).

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

    if isinstance(field, DataSetOptionField):
        json_str = field.to_json()
        if json_str is None:
            return  # Never explicitly set: leave the default instance implicit.
        conf.set(section, ini_key, _escape_percent(json_str), save=False)
    elif isinstance(field, FontOptionField):
        family, size, bold = field.get(sync_env=False)
        conf.set(section, f"{ini_key}_family", family, save=False)
        conf.set(section, f"{ini_key}_size", size, save=False)
        conf.set(section, f"{ini_key}_bold", bold, save=False)
    elif isinstance(field, (ConfigPathOptionField, WorkingDirOptionField)):
        conf.set(section, ini_key, field.get_raw(), save=False)
    elif field_name in DATETIME_FIELDS:
        conf.set(
            section, ini_key, _escape_percent(field.get(sync_env=False)), save=False
        )
    else:
        value = field.get(sync_env=False)
        # ``None`` means "unset": remove any persisted value so that a previously
        # stored non-None value does not linger in the INI (setting a field back
        # to None must clear it, e.g. ``plugins_enabled_list``). This also avoids
        # the INI backend having to coerce None to a numeric type.
        if value is None:
            try:
                conf.remove_option(section, ini_key)
            except Exception:  # pylint: disable=broad-except
                pass
            return
        conf.set(section, ini_key, value, save=False)


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
    options.sync_env()


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
        if field_name in RUNTIME_FIELDS:
            # Runtime IPC value: persisted only by its owner via
            # ``save_runtime_option`` (see :data:`RUNTIME_FIELDS`).
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

    Runtime IPC fields (see :data:`RUNTIME_FIELDS`) are excluded from the bulk
    :func:`save_options_to_ini` so that unrelated saves cannot clobber them.
    Their owner (e.g. the XML-RPC server writing its port) persists them through
    this authoritative single-key write.

    Args:
        options: The DataLab options container.
        name: The option field name to persist.
        conf: The ``UserConfig`` INI backend to write to (defaults to the
         module-level DataLab ``CONF``).
    """
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
    :data:`NON_PERSISTED`).

    Args:
        options: The DataLab options container to inspect.

    Returns:
        Sorted list of uncategorized option field names missing from
         :data:`NON_PERSISTED`.
    """
    from sigima.config import OptionField  # pylint: disable=import-outside-toplevel

    unexpected: list[str] = []
    for name in vars(options):
        if not isinstance(getattr(options, name), OptionField):
            continue
        if options.get_field_category(name):
            continue
        if name in NON_PERSISTED:
            continue
        unexpected.append(name)
    return sorted(unexpected)
