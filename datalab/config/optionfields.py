# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab configuration option fields
-----------------------------------

DataLab-specific :class:`sigima.config.OptionField` subclasses used by the flat,
SigimaX-style configuration container (see :mod:`datalab.config`).

These fields cover behaviours that the generic SigimaX option system does not
provide:

- :class:`ConfigPathOptionField`: a file *basename* stored in the configuration
  directory, resolved to an absolute path on ``get``.
- :class:`WorkingDirOptionField`: a working directory validated on ``set`` and
  returning an empty string on ``get`` when the directory no longer exists.
- :class:`FontOptionField`: a ``(family, size, bold)`` font specification with a
  :meth:`FontOptionField.get_font` helper returning a ``QFont``.
- :class:`DataSetOptionField`: a :class:`guidata.dataset.DataSet` instance, with
  JSON (de)serialization helpers for INI persistence.

All fields whose ``get``/``set`` transform the stored value expose ``get_raw`` /
``set_raw`` accessors returning/accepting the *raw* stored value. The INI<->JSON
converter (see :mod:`datalab.config`) relies on these to avoid the lossy
round-trips that a naive ``get``/``set`` would cause (e.g. a resolved absolute
path being written back where only a basename is expected).
"""

from __future__ import annotations

import json
import os.path as osp
from typing import TYPE_CHECKING, Any

import guidata.dataset as gds
from guidata.configtools import get_family
from sigimax.config import NO_DEFAULT, OptionField
from sigimax.config import FontOptionField as _BaseFontOptionField
from sigimax.utils.conf import Configuration

if TYPE_CHECKING:
    from qtpy import QtGui as QG
    from sigimax.config import AppOptionsContainer


class ConfigPathOptionField(OptionField):
    """Option field for a file stored in the configuration directory.

    The raw stored value is a bare file *basename*. :meth:`get` validates the
    basename and returns the absolute path inside the configuration directory,
    mirroring the historical ``sigimax.utils.conf.ConfigPathOption`` behaviour.

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option.
        default: Default file basename (e.g. ``".DataLab_traceback.log"``).
        description: Description of the option.
    """

    def check(self, value: Any) -> None:
        """Check that the value is a string.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is not a string.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected str, got {type(value).__name__}")

    def get(self, default: Any = NO_DEFAULT, *, sync_env: bool = True) -> str:
        """Return the absolute path inside the configuration directory.

        Args:
            default: Optional basename used when the option is not initialized.
            sync_env: Whether to ensure the environment variable is synchronized
             (keyword-only).

        Returns:
            The absolute path of the file inside the configuration directory.

        Raises:
            ValueError: If the stored value is not a bare basename.
        """
        fname = super().get(default, sync_env=sync_env)
        if osp.basename(fname) != fname:
            raise ValueError(f"Invalid configuration file name {fname}")
        return Configuration.get_path(osp.basename(fname))

    def get_raw(self) -> str:
        """Return the raw stored basename (bypassing path resolution)."""
        return self._value

    def set_raw(self, value: str) -> None:
        """Set the raw stored basename without validation or env sync.

        Args:
            value: The raw basename to store.
        """
        self._value = value
        self.mark_initialized()


class WorkingDirOptionField(OptionField):
    """Option field for a working directory.

    :meth:`set` validates the directory (falling back to its parent when a file
    path is given) and raises when invalid. :meth:`get` returns an empty string
    when the stored directory no longer exists. This mirrors the historical
    ``sigimax.utils.conf.WorkingDirOption`` behaviour.

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option.
        default: Default directory path (empty string by default).
        description: Description of the option.
    """

    def check(self, value: Any) -> None:
        """Check that the value is a string.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is not a string.
        """
        if not isinstance(value, str):
            raise ValueError(f"Expected str, got {type(value).__name__}")

    def get(self, default: Any = NO_DEFAULT, *, sync_env: bool = True) -> str:
        """Return the working directory, or an empty string if it is missing.

        Args:
            default: Optional path used when the option is not initialized.
            sync_env: Whether to ensure the environment variable is synchronized
             (keyword-only).

        Returns:
            The stored directory if it exists, otherwise an empty string.
        """
        path = super().get(default, sync_env=sync_env)
        if osp.isdir(path):
            return path
        return ""

    def set(self, value: str, *, sync_env: bool = True) -> None:
        """Set the working directory, validating that it exists.

        Args:
            value: The directory (or a file whose parent is used) to store.
            sync_env: Whether to synchronize the environment variable
             (keyword-only).

        Raises:
            FileNotFoundError: If neither the value nor its parent is a directory.
        """
        if not osp.isdir(value):
            value = osp.dirname(value)
            if not osp.isdir(value):
                raise FileNotFoundError(f"Invalid working directory name {value}")
        super().set(value, sync_env=sync_env)

    def get_raw(self) -> str:
        """Return the raw stored directory (even if it no longer exists)."""
        return self._value

    def set_raw(self, value: str) -> None:
        """Set the raw stored directory without validation or env sync.

        Args:
            value: The raw directory path to store.
        """
        self._value = value
        self.mark_initialized()


class FontOptionField(_BaseFontOptionField):
    """Option field for a ``(family, size, bold)`` font specification.

    Extends :class:`sigimax.config.FontOptionField` (which normalizes lists to
    tuples on ``set``) to additionally accept a font *family* given as a
    list/tuple of candidate family names (resolved via
    :func:`guidata.configtools.get_family`), and to provide the
    :meth:`get_font` helper returning a ``QFont``.

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option.
        default: Default value as a ``(family, size, bold)`` tuple.
        description: Description of the option.
    """

    def check(self, value: Any) -> None:
        """Check that the value is a valid ``(family, size, bold)`` tuple.

        Relaxes the base check to allow ``family`` to be a list/tuple of
        candidate family names.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is not a valid font specification.
        """
        if value is not None and (
            not isinstance(value, (tuple, list))
            or len(value) != 3
            or not isinstance(value[0], (str, list, tuple))
        ):
            raise ValueError(
                f"Option '{self.name}': expected (family, size, bold) tuple, "
                f"got {value!r}"
            )

    def get_font(self) -> QG.QFont:
        """Return the font as a ``QFont`` instance.

        Returns:
            The configured font as a ``QFont``.
        """
        # Import here to avoid requiring a Qt application when only manipulating
        # configuration files.
        from qtpy import QtGui as QG  # pylint: disable=import-outside-toplevel

        family, size, bold = self.get()
        if isinstance(family, (list, tuple)):
            family = get_family(family)
        return QG.QFont(family, size, QG.QFont.Bold if bold else QG.QFont.Normal)


class DataSetOptionField(OptionField):
    """Option field holding a :class:`guidata.dataset.DataSet` instance.

    The default value is provided through a *default instance*, which may be set
    lazily via :meth:`set_default_instance` (useful when the default depends on
    PlotPy configuration that is not yet available at construction time).

    JSON (de)serialization helpers (:meth:`to_json`, :meth:`from_json`) are used
    by the INI<->JSON converter. Percent-escaping required by ConfigParser is
    handled by the converter, not here.

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option.
        default_instance: Default :class:`~guidata.dataset.DataSet` instance
         (may be ``None`` and set later via :meth:`set_default_instance`).
        description: Description of the option.
    """

    def __init__(
        self,
        container: AppOptionsContainer,
        name: str,
        default_instance: gds.DataSet | None = None,
        description: str = "",
        category: str = "",
    ) -> None:
        self.default_instance = default_instance
        # The actively-set value starts as None; get() falls back to the default
        # instance until an explicit value is assigned.
        super().__init__(
            container, name, default=None, description=description, category=category
        )

    def check(self, value: Any) -> None:
        """Check that the value is a DataSet or None.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is neither a DataSet nor None.
        """
        if value is not None and not isinstance(value, gds.DataSet):
            raise ValueError(
                f"Option '{self.name}': expected a DataSet instance or None, "
                f"got {type(value).__name__}"
            )

    def set_default_instance(self, default_instance: gds.DataSet) -> None:
        """Set the default instance (for lazy initialization).

        Args:
            default_instance: The default DataSet instance to use.
        """
        self.default_instance = default_instance

    def get(
        self, default: Any = NO_DEFAULT, *, sync_env: bool = True
    ) -> gds.DataSet | None:
        """Return the current DataSet instance, or the default instance.

        Args:
            default: Optional DataSet used when the option is not initialized.
            sync_env: Whether to ensure the environment variable is synchronized
             (keyword-only).

        Returns:
            The actively-set DataSet if any, otherwise the default instance.
        """
        value = super().get(default, sync_env=sync_env)
        return value if value is not None else self.default_instance

    def get_raw(self) -> gds.DataSet | None:
        """Return the raw actively-set DataSet (``None`` if never set)."""
        return self._value

    def set_raw(self, value: gds.DataSet | None) -> None:
        """Set the raw DataSet instance without env sync.

        Args:
            value: The DataSet instance to store (or None).
        """
        self._value = value
        self.mark_initialized()

    def to_json(self) -> str | None:
        """Serialize the actively-set DataSet to a JSON string.

        Returns:
            The JSON string of the actively-set DataSet, or ``None`` when no
             value has been explicitly set (so the default instance applies).
        """
        if self._value is None:
            return None
        return gds.dataset_to_json(self._value)

    def from_json(self, json_str: str) -> None:
        """Deserialize a DataSet from a JSON string and store it.

        Args:
            json_str: The JSON string to deserialize.
        """
        data = json.loads(json_str)
        if data.get("class_module") == "datalab.config":
            data["class_module"] = "datalab.config.config"
        self._value = gds.json_to_dataset(json.dumps(data))
        self.mark_initialized()
