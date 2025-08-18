"""
Adapter for Sigima's TableResult, providing features
for storing and retrieving those objects as metadata for DataLab's signal
and image objects.
"""

from __future__ import annotations

from typing import Any, ClassVar, Generator, Union

import numpy as np
from sigima.objects import ImageObj, SignalObj
from sigima.objects.scalar import NO_ROI, TableResult


class TableAdapter:
    """Adapter for TableResult objects.

    This adapter provides a unified interface for working with TableResult objects,
    including metadata storage/retrieval and various data representations.

    Args:
        table: TableResult object to adapt
    """

    # Class constants for metadata storage
    META_PREFIX: ClassVar[str] = "Table_"
    ARRAY_SUFFIX: ClassVar[str] = "_array"
    TITLE_SUFFIX: ClassVar[str] = "_title"
    HEADERS_SUFFIX: ClassVar[str] = "_headers"
    LABELS_SUFFIX: ClassVar[str] = "_labels"

    def __init__(self, table: TableResult) -> None:
        self.table = table
        # Convert TableResult data to the format expected by DataLab
        self._array = self._prepare_array()

    @classmethod
    def from_table_result(cls, table: TableResult) -> TableAdapter:
        """Create TableAdapter from TableResult.

        Args:
            table: TableResult object

        Returns:
            TableAdapter instance
        """
        return cls(table)

    def _prepare_array(self) -> np.ndarray:
        """Convert TableResult data to the format expected by DataLab.

        Returns:
            Array with ROI indices in the first column and values in the following
            columns
        """
        # Create array with ROI indices as the first column
        rows = self.table.data.shape[0]
        cols = self.table.data.shape[1] + 1  # +1 for ROI indices column
        result = np.zeros((rows, cols), dtype=float)

        # Set ROI indices
        if self.table.roi_indices is not None:
            result[:, 0] = self.table.roi_indices
        else:
            result[:, 0] = NO_ROI

        # Set values
        result[:, 1:] = self.table.data

        return result

    @property
    def title(self) -> str:
        """Get the title.

        Returns:
            Title
        """
        return self.table.title

    @property
    def array(self) -> np.ndarray:
        """Get the array with ROI indices and values.

        Returns:
            Array with ROI indices in first column and values in the following columns
        """
        return self._array

    @property
    def headers(self) -> list[str]:
        """Get the column headers.

        Returns:
            Column headers
        """
        return list(self.table.names)

    @property
    def category(self) -> str:
        """Get the category.

        Returns:
            Category (uses the title for backward compatibility)
        """
        return self.table.title

    @property
    def labels(self) -> list[str]:
        """Get the column labels.

        Returns:
            Column labels
        """
        return list(self.table.labels)

    @property
    def raw_data(self) -> np.ndarray:
        """Get raw data (values without ROI indices).

        Returns:
            Array of values (without ROI indices)
        """
        return self._array[:, 1:]

    @property
    def shown_array(self) -> np.ndarray:
        """Get the shown array (same as raw_data for table results).

        Returns:
            Array shown to the user
        """
        return self.raw_data

    @property
    def label_contents(self) -> tuple[tuple[int, str], ...]:
        """Return label contents for compatibility.

        Returns:
            Tuple of couples of (index, text) where index is the column
            and text is the associated label
        """
        return tuple(enumerate(self.headers))

    def set_obj_metadata(self, obj: Union[SignalObj, ImageObj]) -> None:
        """Set object metadata from table result (alias for add_to).

        Args:
            obj: Signal or image object
        """
        self.add_to(obj)

    def add_to(self, obj: Union[SignalObj, ImageObj]) -> None:
        """Add table result to object metadata.

        Args:
            obj: Signal or image object
        """
        # Create a unique key for this result
        base_key = f"{self.META_PREFIX}{self.title}"

        # Store array
        obj.metadata[f"{base_key}{self.ARRAY_SUFFIX}"] = self._array.tolist()

        # Store title
        obj.metadata[f"{base_key}{self.TITLE_SUFFIX}"] = self.title

        # Store headers
        obj.metadata[f"{base_key}{self.HEADERS_SUFFIX}"] = self.headers

        # Store labels
        obj.metadata[f"{base_key}{self.LABELS_SUFFIX}"] = self.labels

        # Store any additional attributes from the TableResult
        if self.table.attrs:
            for key, value in self.table.attrs.items():
                obj.metadata[f"{base_key}_{key}"] = value

    def remove_from(self, obj: Union[SignalObj, ImageObj]) -> None:
        """Remove table result from object metadata.

        Args:
            obj: Signal or image object
        """
        base_key = f"{self.META_PREFIX}{self.title}"

        # Remove standard metadata keys
        keys_to_remove = [
            f"{base_key}{self.ARRAY_SUFFIX}",
            f"{base_key}{self.TITLE_SUFFIX}",
            f"{base_key}{self.HEADERS_SUFFIX}",
            f"{base_key}{self.LABELS_SUFFIX}",
        ]

        # Remove any additional attribute keys
        if self.table.attrs:
            for key in self.table.attrs.keys():
                keys_to_remove.append(f"{base_key}_{key}")

        # Remove all keys that exist in the metadata
        for key in keys_to_remove:
            obj.metadata.pop(key, None)

    @classmethod
    def remove_all_from(cls, obj: Union[SignalObj, ImageObj]) -> None:
        """Remove all table results from object metadata.

        Args:
            obj: Signal or image object
        """
        # Find all table results in the object and remove them
        for adapter in cls.iterate_from_obj(obj):
            adapter.remove_from(obj)

    @classmethod
    def match(cls, key: str, _value: Any) -> bool:
        """Check if the key matches the pattern for a table result.

        Args:
            key: Metadata key
            _value: Metadata value (unused)

        Returns:
            True if the key matches the pattern
        """
        return key.startswith(cls.META_PREFIX) and key.endswith(cls.ARRAY_SUFFIX)

    @classmethod
    def from_metadata_entry(
        cls, obj: Union[SignalObj, ImageObj], key: str
    ) -> TableAdapter:
        """Create a table result adapter from a metadata entry.

        Args:
            obj: Object containing the metadata
            key: Metadata key for the array data

        Returns:
            TableAdapter object

        Raises:
            ValueError: Invalid metadata entry
        """
        if not cls.match(key, obj.metadata[key]):
            raise ValueError(f"Invalid metadata key for table result: {key}")

        base_key = key[: -len(cls.ARRAY_SUFFIX)]
        title = base_key[len(cls.META_PREFIX) :]

        # Parse the metadata entry
        array_data = obj.metadata[key]
        array = np.array(array_data, dtype=float)

        # Get headers and labels
        headers_key = f"{base_key}{cls.HEADERS_SUFFIX}"
        labels_key = f"{base_key}{cls.LABELS_SUFFIX}"

        if headers_key in obj.metadata:
            headers = obj.metadata[headers_key]
        else:
            # For backwards compatibility, create generic headers
            if array.shape[1] > 1:
                headers = [f"Column {i}" for i in range(1, array.shape[1])]
            else:
                headers = ["Value"]

        if labels_key in obj.metadata:
            # Labels not used in the enhanced version, keeping for potential future use
            pass

        # Create TableResult from the data
        if array.size > 0:
            # Extract ROI indices and values
            roi_indices = array[:, 0].astype(int)
            data = array[:, 1:]

            # Create TableResult directly
            table = TableResult(
                title=title,
                names=headers,
                labels=[],  # Labels not used in enhanced version
                data=data,
                roi_indices=roi_indices,
                attrs={},
            )
        else:
            # Create empty TableResult
            table = TableResult(
                title=title,
                names=headers,
                labels=[],
                data=np.zeros((0, len(headers)), dtype=float),
                roi_indices=np.array([], dtype=int),
                attrs={},
            )
        return cls(table)

    @classmethod
    def iterate_from_obj(
        cls, obj: Union[SignalObj, ImageObj]
    ) -> Generator[TableAdapter, None, None]:
        """Iterate over table results stored in an object's metadata.

        Args:
            obj: Signal or image object

        Yields:
            TableAdapter objects
        """
        for key, value in obj.metadata.items():
            if cls.match(key, value):
                try:
                    yield cls.from_metadata_entry(obj, key)
                except (ValueError, TypeError):
                    # Skip invalid entries
                    pass
