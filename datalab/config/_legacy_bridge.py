# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration — transitional legacy bridge
--------------------------------------------------

.. warning::

   **This module is transitional and will be removed at the end of the
   configuration migration.** It exposes a section-style ``Conf`` facade over
   the flat, SigimaX-style :class:`~datalab.config.config_options.DataLabOptions`
   container so that historical ``Conf.<section>.<option>`` call sites keep
   working while they are migrated to the flat API (``Conf.<option>`` +
   :meth:`~sigima.config.OptionField.context`).

The facade maps each legacy ``(section, option)`` pair to a flat option field by
inverting :func:`datalab.config.config_persistence.get_ini_location` (the INI
section is the field *category*; the INI key is the legacy option name). Option
proxies replicate the legacy option API (``get``/``set``/``temp``/``remove`` and
the specialized ``values`` / ``get_font`` / ``set_default_instance`` accessors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sigima.config import OptionField
from sigimax.config import set_conf

from datalab.config.config_options import DataLabOptions
from datalab.config.config_persistence import (
    get_ini_location,
    load_options_from_ini,
)
from datalab.config.optionfields import (
    ConfigPathOptionField,
    DataSetOptionField,
    WorkingDirOptionField,
)
from datalab.utils import conf as _confmod
from datalab.utils.conf import Configuration

if TYPE_CHECKING:
    import guidata.dataset as gds

_UNSET = object()


class _OptionProxy:
    """Legacy-compatible proxy around a flat :class:`OptionField`.

    Args:
        container: The flat options container the field belongs to.
        field: The wrapped option field.
    """

    def __init__(self, container: DataLabOptions, field: OptionField) -> None:
        self._container = container
        self._field = field

    def get(self, default: Any = _UNSET) -> Any:
        """Return the option value, emulating the legacy ``get(default)``.

        Legacy ``Option.get(default)`` returned the persisted value when the
        option had been set, and the caller-provided ``default`` otherwise. The
        flat field always carries a value (its own default), so "unset" is
        approximated as "the field is still equal to its default": in that case
        the caller-provided ``default`` is returned. This preserves call sites
        that pass a meaningful fallback different from the field default (e.g.
        axis format strings ``sig_format``/``ima_format`` or ``show_label``).

        The substitution is skipped for the specialized DataLab field types
        (config-path, working-directory, DataSet) whose ``get`` returns a
        transformed value and which handle their own defaults.

        Args:
            default: Optional legacy fallback value.

        Returns:
            The current option value, or ``default`` when the field is still at
             its default value.
        """
        value = self._field.get()
        if default is not _UNSET and not isinstance(
            self._field,
            (DataSetOptionField, ConfigPathOptionField, WorkingDirOptionField),
        ):
            field_default = self._container.get_default_raw(self._field.name)
            if value == field_default:
                return default
        return value

    def set(self, value: Any) -> None:
        """Set the option value.

        Args:
            value: The new value to assign.
        """
        self._field.set(value)

    def temp(self, value: Any):
        """Temporarily override the option (legacy alias for ``context``).

        Args:
            value: Temporary value used within the context.

        Returns:
            A context manager restoring the previous value on exit.
        """
        return self._field.context(value)

    def context(self, value: Any):
        """Temporarily override the option (target flat API).

        Args:
            value: Temporary value used within the context.

        Returns:
            A context manager restoring the previous value on exit.
        """
        return self._field.context(value)

    def remove(self) -> None:
        """Reset the option to its default and drop it from the INI backend."""
        location = get_ini_location(self._container, self._field.name)
        if location is not None:
            section, ini_key = location
            try:
                _confmod.CONF.remove_option(section, ini_key)
            except Exception:  # pylint: disable=broad-except
                pass
        default = self._container.get_default_raw(self._field.name)
        try:
            self._field.set(default)
        except Exception:  # pylint: disable=broad-except
            pass

    @property
    def values(self) -> list[str]:
        """Return the allowed values of an enum option (legacy alias)."""
        return self._field.choices

    @property
    def option(self) -> str:
        """Return the legacy INI option key of the field."""
        location = get_ini_location(self._container, self._field.name)
        return location[1] if location is not None else self._field.name

    @property
    def section(self) -> str:
        """Return the legacy INI section (the field category)."""
        location = get_ini_location(self._container, self._field.name)
        return location[0] if location is not None else ""

    def __getattr__(self, item: str) -> Any:
        """Delegate any other attribute access to the wrapped field.

        This exposes specialized accessors such as ``get_font`` (font fields) and
        ``set_default_instance`` / ``to_json`` / ``from_json`` (DataSet fields).

        Args:
            item: Attribute name.

        Returns:
            The delegated attribute from the wrapped field.
        """
        return getattr(self._field, item)


class _SectionProxy:
    """Legacy-compatible proxy exposing a category's option fields.

    Args:
        container: The flat options container.
        section: The legacy section name (equal to the field category).
        keymap: Mapping of legacy option names to flat field names.
    """

    def __init__(
        self, container: DataLabOptions, section: str, keymap: dict[str, str]
    ) -> None:
        object.__setattr__(self, "_container", container)
        object.__setattr__(self, "_section", section)
        object.__setattr__(self, "_keymap", keymap)
        # Cache one proxy per option name so that repeated accesses return the
        # same object. This is required for monkeypatching to work (e.g.
        # ``patch("datalab.config.Conf.main.plugins_enabled.get")``), which sets
        # an attribute on the resolved proxy instance.
        object.__setattr__(self, "_proxies", {})

    def get_name(self) -> str:
        """Return the legacy section name (equal to the field category)."""
        return object.__getattribute__(self, "_section")

    def __getattr__(self, name: str) -> _OptionProxy:
        keymap = object.__getattribute__(self, "_keymap")
        flat_name = keymap.get(name)
        if flat_name is None:
            raise AttributeError(
                f"Unknown configuration option '{name}' in section "
                f"'{object.__getattribute__(self, '_section')}'"
            )
        proxies = object.__getattribute__(self, "_proxies")
        proxy = proxies.get(name)
        if proxy is None:
            container = object.__getattribute__(self, "_container")
            proxy = _OptionProxy(container, getattr(container, flat_name))
            proxies[name] = proxy
        return proxy


class _ViewSectionProxy(_SectionProxy):
    """View section proxy with the historical ``*_def_*`` dictionary helpers."""

    def get_def_dict(self, category: str) -> dict[str, Any]:
        """Return default visualization settings as a dictionary.

        Args:
            category: ``"ima"`` or ``"sig"``.

        Returns:
            Mapping of setting name (without the ``<category>_def_`` prefix) to
             its current value.
        """
        assert category in ("ima", "sig")
        container = object.__getattribute__(self, "_container")
        prefix = f"{category}_def_"
        def_dict: dict[str, Any] = {}
        for name in vars(container):
            if not name.startswith(prefix):
                continue
            field = getattr(container, name)
            if not isinstance(field, OptionField):
                continue
            value = field.get()
            if value is not None:
                def_dict[name[len(prefix) :]] = value
        return def_dict

    def set_def_dict(self, category: str, def_dict: dict[str, Any]) -> None:
        """Set default visualization settings from a dictionary.

        Args:
            category: ``"ima"`` or ``"sig"``.
            def_dict: Mapping of setting name (without prefix) to value.
        """
        assert category in ("ima", "sig")
        container = object.__getattribute__(self, "_container")
        prefix = f"{category}_def_"
        for name in vars(container):
            if not name.startswith(prefix):
                continue
            field = getattr(container, name)
            if not isinstance(field, OptionField):
                continue
            key = name[len(prefix) :]
            if key in def_dict:
                field.set(def_dict[key])


class _ConfFacade:
    """Section-style facade over the flat :class:`DataLabOptions` container."""

    def __init__(self) -> None:
        self._options = DataLabOptions()
        # Install the DataLab options as the active SigimaX configuration so that
        # reused SigimaX widgets/adapters (which read ``sigimax.config.get_conf()``)
        # operate on the very same container as DataLab's ``Conf``.
        set_conf(self._options)
        self._sections: dict[str, _SectionProxy] = {}
        self._flat_proxies: dict[str, _OptionProxy] = {}
        self._build_sections()

    def _build_sections(self) -> None:
        """Build the ``(section -> option -> flat_name)`` mapping and proxies."""
        keymaps: dict[str, dict[str, str]] = {}
        for name in vars(self._options):
            field = getattr(self._options, name)
            if not isinstance(field, OptionField):
                continue
            location = get_ini_location(self._options, name)
            if location is None:
                continue
            section, ini_key = location
            keymaps.setdefault(section, {})[ini_key] = name
        for section, keymap in keymaps.items():
            proxy_cls = _ViewSectionProxy if section == "view" else _SectionProxy
            self._sections[section] = proxy_cls(self._options, section, keymap)

    # -- Section access --------------------------------------------------

    def __getattr__(self, name: str) -> _SectionProxy:
        sections = object.__getattribute__(self, "_sections")
        if name in sections:
            return sections[name]
        # Flat option access (target API for migrated call sites):
        # ``Conf.<flat_option>`` returns a cached option proxy so that migrated
        # call sites can use ``Conf.color_mode.get()`` / ``.context(...)`` while
        # the transitional bridge is still in place.
        options = object.__getattribute__(self, "_options")
        field = getattr(options, name, None)
        if isinstance(field, OptionField):
            flat_proxies = object.__getattribute__(self, "_flat_proxies")
            proxy = flat_proxies.get(name)
            if proxy is None:
                proxy = _OptionProxy(options, field)
                flat_proxies[name] = proxy
            return proxy
        raise AttributeError(f"Unknown configuration section or option '{name}'")

    # -- Flat container access (for the migrated call sites) -------------

    @property
    def options(self) -> DataLabOptions:
        """Return the underlying flat options container."""
        return self._options

    # -- Backend / lifecycle (legacy Configuration API) -----------------

    def initialize(self, name: str, version: str, load: bool) -> None:
        """Initialize the INI backend and load options into the container.

        Args:
            name: Configuration application name.
            version: Configuration version string.
            load: Whether to load persisted values from the INI file.
        """
        self._options.set_ini_persist_enabled(False)
        Configuration.initialize(name, version, load)
        if load:
            load_options_from_ini(self._options, _confmod.CONF)
        self._options.set_ini_persist_enabled(True)

    def reset(self) -> None:
        """Reset the INI backend and restore option defaults."""
        self._options.set_ini_persist_enabled(False)
        Configuration.reset()
        self._options.reset_to_defaults()

    def reload_from_ini(self) -> None:
        """Reload option values from the INI backend (app session start).

        Models an application startup reading persisted settings: the flat
        container is refreshed from the current INI file without triggering
        write-through during the load.
        """
        self._options.set_ini_persist_enabled(False)
        load_options_from_ini(self._options, _confmod.CONF)
        self._options.set_ini_persist_enabled(True)

    def get_path(self, basename: str) -> str:
        """Return a path inside the configuration directory.

        Args:
            basename: File base name.

        Returns:
            The absolute path inside the configuration directory.
        """
        return Configuration.get_path(basename)

    def get_filename(self) -> str:
        """Return the configuration file name."""
        return Configuration.get_filename()

    def to_dict(self) -> dict:
        """Return the INI backend configuration as a dictionary."""
        return Configuration.to_dict()

    def set_default_instance(self, name: str, instance: gds.DataSet) -> None:
        """Set the default DataSet instance of a flat option field.

        Args:
            name: Flat option field name.
            instance: Default DataSet instance.
        """
        getattr(self._options, name).set_default_instance(instance)


#: Singleton legacy-compatible configuration facade.
Conf = _ConfFacade()  # pylint: disable=invalid-name
