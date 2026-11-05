# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Native HDF5 I/O module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import contextlib
import importlib
import os
import os.path as osp
import stat
import tempfile
from collections.abc import Generator
from typing import Any, Callable

from guidata.io import HDF5Reader, HDF5Writer
from guidata.io.h5fmt import NoDefault
from packaging.version import InvalidVersion, Version

import datalab

DATALAB_VERSION_NAME = "DataLab_Version"
DATALAB_PACKAGE_NAME = "datalab"

H5_CALLABLE_PREFIX = "#callable#"

#: First DataLab version which no longer stores automatic curve styles in signal
#: metadata (earlier versions stored them, see :func:`is_legacy_style_version`)
NON_PERSISTENT_CURVE_STYLES_VERSION = (1, 4)


def is_legacy_style_version(version: str | None) -> bool:
    """Return True if a file saved by the given DataLab version may store automatic
    curve styles in signal metadata (DataLab 1.3 or earlier)

    Args:
        version: DataLab version that saved the file, as stored in the file

    Returns:
        True if the version is earlier than 1.4 (pre-releases of 1.4 included in
        1.4); False if it is 1.4 or later, missing or invalid
    """
    if not isinstance(version, str):
        return False
    try:
        release = Version(version).release
    except InvalidVersion:
        return False
    return release < NON_PERSISTENT_CURVE_STYLES_VERSION


class NativeH5Writer(HDF5Writer):
    """DataLab signal/image objects HDF5 guidata Dataset Writer class

    Args:
        filename (str): HDF5 file name
    """

    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.h5[DATALAB_VERSION_NAME] = datalab.__version__

    @staticmethod
    def serialize_func_or_class(obj: Callable | type) -> str:
        """Serialize a function or a class object

        Args:
            obj: Object to serialize

        Returns:
            str: Serialized object
        """
        if not obj.__module__.startswith(DATALAB_PACKAGE_NAME):
            raise ValueError(
                f"Only {DATALAB_PACKAGE_NAME} functions and classes can be serialized"
            )
        val = f"{H5_CALLABLE_PREFIX}{obj.__module__}."
        if isinstance(obj, type):
            return val + obj.__name__
        return val + obj.__qualname__

    # Reimplement the write method to handle callable objects
    def write(self, val: Any, group_name: str | None = None) -> None:
        """
        Write a value depending on its type, optionally within a named group.

        Args:
            val: The value to be written.
            group_name: The name of the group. If provided, the group
             context will be used for writing the value.
        """
        try:
            super().write(val, group_name)
        except NotImplementedError:
            if callable(val):
                super().write_str(self.serialize_func_or_class(val))
                if group_name:
                    self.end(group_name)
            else:
                raise


def set_final_permissions(tmpname: str, filename: str) -> None:
    """Give the temporary file the permissions the final file must have.

    Args:
        tmpname: Temporary file name (created with owner-only permissions).
        filename: Final file name: its permissions are kept if it exists,
         otherwise those of a newly created file are applied.
    """
    if osp.exists(filename):
        mode = stat.S_IMODE(os.stat(filename).st_mode)
    else:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    os.chmod(tmpname, mode)


@contextlib.contextmanager
def atomic_native_writer(filename: str) -> Generator[NativeH5Writer, None, None]:
    """Write a native DataLab HDF5 file without losing the previous one on failure.

    The file is written to a temporary file of the same directory, which replaces
    ``filename`` only once it has been completely written and closed. If writing
    fails (serialization error, full disk, interruption...), the temporary file is
    removed and the previous file, if any, is left untouched.

    Args:
        filename: HDF5 file name

    Yields:
        Writer of the temporary file
    """
    filename = osp.abspath(filename)
    fd, tmpname = tempfile.mkstemp(
        prefix=f".{osp.basename(filename)}.", suffix=".tmp", dir=osp.dirname(filename)
    )
    os.close(fd)
    try:
        writer = NativeH5Writer(tmpname)
        try:
            yield writer
        finally:
            writer.close()
        set_final_permissions(tmpname, filename)
        os.replace(tmpname, filename)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmpname)
        raise


class NativeH5Reader(HDF5Reader):
    """DataLab signal/image objects HDF5 guidata dataset Writer class

    Args:
        filename (str): HDF5 file name
    """

    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.version = self.h5[DATALAB_VERSION_NAME]
        self.__saved_version: str | None = None
        self.__saved_version_read = False

    @property
    def saved_version(self) -> str | None:
        """DataLab version that saved the file, or None if it is not readable"""
        if not self.__saved_version_read:
            value = self.version[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            self.__saved_version = value if isinstance(value, str) else None
            self.__saved_version_read = True
        return self.__saved_version

    @property
    def has_legacy_curve_styles(self) -> bool:
        """True if the file was saved by a DataLab version which stored automatic
        curve styles in signal metadata (DataLab 1.3 or earlier)"""
        return is_legacy_style_version(self.saved_version)

    @staticmethod
    def deserialize_func_or_class(obj: str) -> Callable | type:
        """Deserialize a function or a class object

        Args:
            obj: Serialized object

        Returns:
            Callable | type: Deserialized object
        """
        parts = obj[len(H5_CALLABLE_PREFIX) :].split(".")
        if not parts or not parts[0].startswith(DATALAB_PACKAGE_NAME):
            raise ValueError(
                f"Only {DATALAB_PACKAGE_NAME} functions and classes can be deserialized"
            )
        # Walk path parts: find longest valid module prefix, then resolve the
        # remaining attribute chain (supports methods like ``Class.method``).
        for split_index in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:split_index])
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            attr: Any = module
            try:
                for name in parts[split_index:]:
                    attr = getattr(attr, name)
            except AttributeError:
                continue
            return attr
        raise ImportError(f"Cannot deserialize callable: {obj}")

    # Reimplement the read method to handle callable objects
    def read(
        self,
        group_name: str | None = None,
        func: Callable[[], Any] | None = None,
        instance: Any | None = None,
        default: Any | NoDefault = NoDefault,
    ) -> Any:
        """
        Read a value from the current group or specified group_name.

        Args:
            group_name: The name of the group to read from. Defaults to None.
            func: The function to use for reading the value. Defaults to None.
            instance: An object that implements the DataSet-like `deserialize` method.
             Defaults to None.
            default: The default value to return if the value is not found.
             Defaults to `NoDefault` (no default value: raises an exception if the
             value is not found).

        Returns:
            The read value.
        """
        val = super().read(group_name, func=func, instance=instance, default=default)
        if isinstance(val, str) and val.startswith(H5_CALLABLE_PREFIX):
            return self.deserialize_func_or_class(val)
        return val
