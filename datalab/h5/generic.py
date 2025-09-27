# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Generic HDF5 format support
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import h5py
import numpy as np
from guidata.utils.misc import to_string
from sigima.objects import create_image, create_signal

from datalab.h5 import common, utils

# =============================================================================
# Encoding and Data Reading Utilities
# =============================================================================


def safe_decode_bytes(data, fallback="<binary data>"):
    """Safely decode bytes to string using multiple encoding strategies."""
    if isinstance(data, str):
        return data
    if not isinstance(data, bytes):
        return str(data)

    # Try encodings in order of preference
    for encoding in ["utf-8", "latin1", "cp1252", "iso-8859-1", "ascii"]:
        try:
            decoded = data.decode(encoding)
            # For legacy encodings, validate the result looks reasonable
            if encoding in ["latin1", "cp1252", "iso-8859-1"]:
                if _is_reasonable_text(decoded):
                    return decoded
            else:
                return decoded
        except (UnicodeDecodeError, UnicodeError):
            continue

    # Final fallback with replacement characters
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return fallback


def _is_reasonable_text(text):
    """Check if decoded text looks reasonable (mostly printable)."""
    if not text:
        return True

    printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
    ratio = printable_chars / len(text)

    # Accept if mostly printable or short enough to be likely text
    return ratio >= 0.8 or len(text) < 20


def safe_read_dataset(dset, fallback_data=None):
    """Safely read HDF5 dataset with encoding error handling."""
    try:
        return dset[()]
    except UnicodeDecodeError:
        # Try alternative reading strategies for problematic encodings
        return _try_alternative_read(dset, fallback_data)
    except (TypeError, ValueError, OSError):
        return fallback_data


def _try_alternative_read(dset, fallback_data):
    """Try alternative strategies to read datasets with encoding issues."""
    strategies = [
        lambda d: d.asstr()[()],  # Try string conversion
        lambda d: d.astype("S")[()],  # Try reading as bytes
    ]

    for strategy in strategies:
        try:
            return strategy(dset)
        except Exception:
            continue

    return fallback_data


# =============================================================================
# Text Formatting Utilities
# =============================================================================


def format_text_data(data):
    """Format various types of data for text display."""
    if data is None:
        return "<unreadable data>"

    try:
        return to_string(data)
    except (UnicodeDecodeError, UnicodeError):
        return _handle_encoding_issues(data)


def _handle_encoding_issues(data):
    """Handle data with encoding issues."""
    if isinstance(data, bytes):
        return safe_decode_bytes(data)

    if isinstance(data, np.ndarray):
        if data.dtype.kind in ["S", "a", "U"]:  # String arrays
            return _format_string_array(data)
        elif data.dtype.names:  # Compound data
            return _format_compound_data(data)

    return f"<data with encoding issues: {type(data)}>"


def _format_string_array(data):
    """Format string arrays with encoding handling."""
    try:
        if data.size == 1:
            return safe_decode_bytes(data.item())
        else:
            # Show first few elements
            items = []
            for i, item in enumerate(data.flat):
                if i >= 5:
                    items.append("...")
                    break
                items.append(safe_decode_bytes(item))
            return f"[{', '.join(items)}]"
    except Exception:
        return f"<string array: {data.shape} {data.dtype}>"


def _format_compound_data(data):
    """Format compound data with encoding handling."""
    try:
        result_parts = []
        for field_name in data.dtype.names:
            field_data = data[field_name]
            if hasattr(field_data, "item"):
                field_data = field_data.item()

            if isinstance(field_data, bytes):
                field_value = safe_decode_bytes(field_data)
            else:
                field_value = str(field_data)

            result_parts.append(f"{field_name}: {field_value}")

        return f"({', '.join(result_parts)})"
    except Exception:
        return f"<compound data: {data.dtype}>"


# =============================================================================
# Base Node Class
# =============================================================================


class BaseGenericNode(common.BaseNode):
    """Base class for generic HDF5 data nodes with encoding support."""

    @classmethod
    def match(cls, dset):
        """Return True if h5 dataset matches this node pattern."""
        return not isinstance(dset, h5py.Group)

    @property
    def icon_name(self):
        """Icon name associated to node."""
        return "h5scalar.svg"

    @property
    def data(self):
        """Data associated to node, if available."""
        return safe_read_dataset(self.dset, fallback_data=None)

    @property
    def dtype_str(self):
        """Return string representation of node data type."""
        try:
            return str(self.dset.dtype)
        except (UnicodeDecodeError, TypeError, ValueError):
            if self.data is None:
                return "unknown"
            try:
                return str(self.data.dtype)
            except Exception:
                return "unknown"

    @property
    def text(self):
        """Return node textual representation."""
        return format_text_data(self.data)


# =============================================================================
# Specialized Node Classes
# =============================================================================


class GenericScalarNode(BaseGenericNode):
    """Node for scalar HDF5 data."""

    @classmethod
    def match(cls, dset):
        """Match scalar numeric data."""
        if not super().match(dset):
            return False
        data = safe_read_dataset(dset)
        return (
            data is not None
            and isinstance(data, np.generic)
            and utils.is_supported_num_dtype(data)
        )


class GenericTextNode(BaseGenericNode):
    """Node for text/string HDF5 data."""

    @classmethod
    def match(cls, dset):
        """Match text or string data."""
        if not super().match(dset):
            return False
        data = safe_read_dataset(dset)
        if data is None:
            # Try to match based on dtype for unreadable data
            try:
                dtype = dset.dtype
                return dtype.kind in ["S", "a", "U"] or "str" in str(dtype)
            except Exception:
                return False
        return isinstance(data, bytes) or utils.is_supported_str_dtype(data)

    @property
    def dtype_str(self):
        """Return simplified dtype for text data."""
        return "string"

    @property
    def text(self):
        """Return formatted text with special handling for single arrays."""
        if self.data is None:
            return "<unreadable data>"

        try:
            if utils.is_single_str_array(self.data):
                item = self.data[0]
                return safe_decode_bytes(item) if isinstance(item, bytes) else str(item)
            return format_text_data(self.data)
        except (UnicodeDecodeError, UnicodeError):
            return _handle_text_encoding_issues(self.data)


def _handle_text_encoding_issues(data):
    """Handle encoding issues specific to text nodes."""
    if isinstance(data, bytes):
        return safe_decode_bytes(data)
    elif isinstance(data, np.ndarray) and data.dtype.kind in ["S", "a"]:
        try:
            if data.size == 1:
                return safe_decode_bytes(data.item())
            else:
                decoded = [safe_decode_bytes(item) for item in data.flat]
                return str(decoded[:10])  # Show first 10 elements
        except Exception:
            return f"<string data: {data.shape}>"
    return "<text with encoding issues>"


class GenericArrayNode(BaseGenericNode):
    """Node for array HDF5 data, including numeric arrays from compound data."""

    IS_ARRAY = True

    @classmethod
    def match(cls, dset):
        """Match numeric array data, including convertible compound data."""
        if not super().match(dset):
            return False
        data = safe_read_dataset(dset)

        if data is None:
            return False

        # First check direct numeric arrays
        if (
            utils.is_supported_num_dtype(data)
            and isinstance(data, np.ndarray)
            and len(data.shape) in (1, 2)
        ):
            return True

        # Then check compound data that can be converted to numeric arrays
        if (
            isinstance(data, np.ndarray)
            and hasattr(data.dtype, "names")
            and data.dtype.names is not None
        ):
            return cls._can_convert_compound_to_numeric(data)

        return False

    @classmethod
    def _can_convert_compound_to_numeric(cls, data):
        """Check if compound data can be converted to a supported numeric array."""
        try:
            numeric_array = cls._extract_numeric_from_compound(data)
            return (
                numeric_array is not None
                and utils.is_supported_num_dtype(numeric_array)
                and isinstance(numeric_array, np.ndarray)
                and len(numeric_array.shape) in (1, 2)
            )
        except Exception:
            return False

    @classmethod
    def _extract_numeric_from_compound(cls, data):
        """Extract a numeric array from compound data."""
        if not (hasattr(data.dtype, "names") and data.dtype.names):
            return None

        # Find ALL fields and check if they are numeric
        all_fields = list(data.dtype.names)
        numeric_fields = []
        for field_name in all_fields:
            field_dtype = data.dtype.fields[field_name][0]
            if np.issubdtype(field_dtype, np.number):
                numeric_fields.append(field_name)

        # Only convert if ALL fields are numeric (preserve all information)
        # or if there's a single numeric field and no important string data
        if len(numeric_fields) == 0:
            return None

        if len(numeric_fields) != len(all_fields):
            # Mixed data - check if non-numeric fields contain meaningful data
            for field_name in all_fields:
                if field_name not in numeric_fields:
                    field_data = data[field_name]
                    # If there's meaningful string data, don't convert
                    if cls._has_meaningful_string_data(field_data):
                        return None

        try:
            # If single numeric field, extract it directly
            if len(numeric_fields) == 1:
                return data[numeric_fields[0]]

            # Multiple numeric fields: stack them if compatible shapes
            field_data = [data[field] for field in numeric_fields]

            # Check if all fields have the same shape
            shapes = [arr.shape for arr in field_data]
            if len(set(shapes)) == 1:
                # Stack along new axis to create 2D array
                return np.stack(field_data, axis=-1)

            return None  # Can't easily convert mixed shapes

        except Exception:
            return None

    @classmethod
    def _has_meaningful_string_data(cls, field_data):
        """Check if string field contains meaningful data worth preserving."""
        try:
            if hasattr(field_data, "flat"):
                # Check if most entries are non-empty and meaningful
                non_empty_count = 0
                for item in field_data.flat:
                    if isinstance(item, bytes):
                        decoded = item.decode("utf-8", errors="ignore").strip()
                        if decoded and len(decoded) > 0:
                            non_empty_count += 1
                    elif isinstance(item, str) and item.strip():
                        non_empty_count += 1

                # If most entries have meaningful content, preserve it
                return non_empty_count / field_data.size > 0.5
            return True  # Default to preserving unknown string data
        except Exception:
            return True  # Conservative: preserve if we can't determine

    @property
    def data(self):
        """Data associated to node, if available."""
        raw_data = safe_read_dataset(self.dset, fallback_data=None)

        # If this is compound data, try to extract numeric array
        if (
            raw_data is not None
            and isinstance(raw_data, np.ndarray)
            and hasattr(raw_data.dtype, "names")
            and raw_data.dtype.names is not None
        ):
            numeric_data = self._extract_numeric_from_compound(raw_data)
            if numeric_data is not None:
                return numeric_data

        return raw_data

    def is_supported(self) -> bool:
        """Return True if node is associated to supported data"""
        return self.data.size > 1

    @property
    def __is_signal(self):
        """Return True if array represents a signal"""
        shape = self.data.shape
        return len(shape) == 1 or shape[0] in (1, 2) or shape[1] in (1, 2)

    @property
    def icon_name(self):
        """Icon name associated to node"""
        if self.is_supported():
            return "signal.svg" if self.__is_signal else "image.svg"
        return "h5array.svg"

    @property
    def shape_str(self):
        """Return string representation of node shape, if any"""
        return " x ".join([str(size) for size in self.data.shape])

    @property
    def dtype_str(self):
        """Return string representation of node data type, if any"""
        return str(self.data.dtype)

    @property
    def text(self):
        """Return node textual representation"""
        return str(self.data)

    def create_native_object(self):
        """Create native object, if supported"""
        if self.__is_signal:
            obj = create_signal(self.object_title)
            try:
                self.set_signal_data(obj)
            except ValueError:
                obj = None
        else:
            obj = create_image(self.object_title)
            try:
                self.set_image_data(obj)
            except ValueError:
                obj = None
        return obj


class GenericCompoundNode(BaseGenericNode):
    """Node for compound/structured HDF5 data that can't convert to numeric arrays."""

    IS_ARRAY = True

    @classmethod
    def match(cls, dset):
        """Match compound/structured data that cannot be converted to numeric arrays."""
        if not super().match(dset):
            return False

        data = safe_read_dataset(dset)
        if data is None:
            # Try to match based on dtype if we can't read the data
            try:
                return dset.dtype.names is not None
            except Exception:
                return False

        # Check if it's compound data (structured array)
        if not (
            isinstance(data, np.ndarray)
            and hasattr(data.dtype, "names")
            and data.dtype.names is not None
        ):
            return False

        # IMPORTANT: Only match if GenericArrayNode cannot handle this data
        # Try to convert to a numeric array first
        if cls._can_convert_to_numeric_array(data):
            return False  # Let GenericArrayNode handle it

        return True  # We handle compound data that can't be converted

    @classmethod
    def _can_convert_to_numeric_array(cls, data):
        """Check if compound data can be converted to a numeric array."""
        try:
            # Try to extract numeric fields and create a pure numeric array
            numeric_array = cls._extract_numeric_array(data)
            if numeric_array is None:
                return False

            # Check if the resulting array would be supported by GenericArrayNode
            return (
                utils.is_supported_num_dtype(numeric_array)
                and isinstance(numeric_array, np.ndarray)
                and len(numeric_array.shape) in (1, 2)
            )
        except Exception:
            return False

    @classmethod
    def _extract_numeric_array(cls, data):
        """Try to extract a pure numeric array from compound data."""
        # Use the same logic as GenericArrayNode
        return GenericArrayNode._extract_numeric_from_compound(data)

    @property
    def dtype_str(self):
        """Return detailed compound dtype information."""
        try:
            dtype = self.dset.dtype
            if dtype.names:
                field_info = []
                for name in dtype.names:
                    field_dtype = dtype.fields[name][0]
                    field_info.append(f"{name}: {field_dtype}")
                return f"compound({', '.join(field_info)})"
            return str(dtype)
        except Exception:
            return super().dtype_str

    @property
    def text(self):
        """Return formatted compound data."""
        if self.data is None:
            return "<unreadable compound data>"

        try:
            return self._format_compound_data()
        except Exception:
            return f"<compound data: {self.data.shape} records>"

    def _format_compound_data(self):
        """Format compound data for display."""
        if not (hasattr(self.data.dtype, "names") and self.data.dtype.names):
            return super().text

        if self.data.size == 1:
            return self._format_single_record()
        else:
            return self._format_multiple_records()

    def _format_single_record(self):
        """Format a single compound record."""
        parts = []
        for field_name in self.data.dtype.names:
            field_value = self.data[field_name].item()
            if isinstance(field_value, bytes):
                field_value = safe_decode_bytes(field_value)
            parts.append(f"{field_name}: {field_value}")
        return f"({', '.join(parts)})"

    def _format_multiple_records(self):
        """Format multiple compound records."""
        records = []
        for i, record in enumerate(self.data.flat):
            if i >= 3:  # Show max 3 records
                records.append("...")
                break

            parts = []
            for field_name in self.data.dtype.names:
                field_value = record[field_name]
                if isinstance(field_value, bytes):
                    field_value = safe_decode_bytes(field_value)
                parts.append(f"{field_name}: {field_value}")
            records.append(f"({', '.join(parts)})")

        return f"[{', '.join(records)}]"


# =============================================================================
# Node Registration
# =============================================================================

# Register all node types with the factory
common.NODE_FACTORY.register(GenericScalarNode, is_generic=True)
common.NODE_FACTORY.register(GenericTextNode, is_generic=True)
common.NODE_FACTORY.register(GenericArrayNode, is_generic=True)
common.NODE_FACTORY.register(GenericCompoundNode, is_generic=True)
