# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Adapter for Sigima's TableResult, providing features
for storing and retrieving those objects as metadata for DataLab's signal
and image objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generator, Union

from sigima.objects import ImageObj, SignalObj
from sigima.objects.scalar import NO_ROI, TableResult

if TYPE_CHECKING:
    import pandas as pd


class TableAdapter:
    """Adapter for TableResult objects.

    This adapter provides a unified interface for working with TableResult objects,
    including metadata storage/retrieval and various data representations.

    Args:
        table: TableResult object to adapt
    """

    # Class constants for metadata storage
    META_PREFIX: ClassVar[str] = "Table_"
    DATA_SUFFIX: ClassVar[str] = "_data"

    def __init__(self, table: TableResult) -> None:
        self.table = table

    @classmethod
    def from_table_result(cls, table: TableResult) -> TableAdapter:
        """Create TableAdapter from TableResult.

        Args:
            table: TableResult object

        Returns:
            TableAdapter instance
        """
        return cls(table)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the table result to a pandas DataFrame.

        Returns:
            DataFrame with columns as in self.headers, and optional 'roi_index' column.
        """
        return self.table.to_dataframe()

    @property
    def title(self) -> str:
        """Get the title.

        Returns:
            Title
        """
        return self.table.title

    @property
    def headers(self) -> list[str]:
        """Get the column headers.

        Returns:
            Column headers
        """
        return list(self.table.headers)

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
    def label_contents(self) -> tuple[tuple[int, str], ...]:
        """Return label contents for compatibility.

        Returns:
            Tuple of couples of (index, text) where index is the column
            and text is the associated label
        """
        return tuple(enumerate(self.headers))

    # DataFrame-based helper methods for modern data access
    def get_roi_data(self, roi_index: int) -> "pd.DataFrame":
        """Get data for a specific ROI using DataFrame operations.

        Args:
            roi_index: ROI index to filter by (use NO_ROI for non-ROI data)

        Returns:
            DataFrame containing only data for the specified ROI
        """
        df = self.to_dataframe()
        if "roi_index" in df.columns:
            return df[df["roi_index"] == roi_index].drop(columns=["roi_index"])
        return df

    def get_column_values(self, column_name: str, roi_index: int = None) -> list:
        """Get values for a specific column, optionally filtered by ROI.

        Args:
            column_name: Name of the column to retrieve
            roi_index: Optional ROI index to filter by

        Returns:
            List of values for the specified column
        """
        df = self.to_dataframe()
        if roi_index is not None and "roi_index" in df.columns:
            df = df[df["roi_index"] == roi_index]
        return df[column_name].tolist()

    def get_unique_roi_indices(self) -> list[int]:
        """Get unique ROI indices present in the data.

        Returns:
            List of unique ROI indices
        """
        df = self.to_dataframe()
        if "roi_index" in df.columns:
            return sorted(df["roi_index"].unique().tolist())
        return [NO_ROI] if len(df) > 0 else []

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
        key = f"{self.META_PREFIX}{self.title}{self.DATA_SUFFIX}"

        # Store the complete TableResult as a dictionary for efficient recovery
        obj.metadata[key] = self.table.to_dict()

    def remove_from(self, obj: Union[SignalObj, ImageObj]) -> None:
        """Remove table result from object metadata.

        Args:
            obj: Signal or image object
        """
        key = f"{self.META_PREFIX}{self.title}{self.DATA_SUFFIX}"
        obj.metadata.pop(key, None)

    @classmethod
    def remove_all_from(cls, obj: Union[SignalObj, ImageObj]) -> None:
        """Remove all table results from object metadata.

        Args:
            obj: Signal or image object
        """
        # Find all table results in the object and remove them
        for adapter in list(cls.iterate_from_obj(obj)):
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
        return key.startswith(cls.META_PREFIX) and key.endswith(cls.DATA_SUFFIX)

    @classmethod
    def from_metadata_entry(
        cls, obj: Union[SignalObj, ImageObj], key: str
    ) -> TableAdapter:
        """Create a table result adapter from a metadata entry.

        Args:
            obj: Object containing the metadata
            key: Metadata key for the table data

        Returns:
            TableAdapter object

        Raises:
            ValueError: Invalid metadata entry
        """
        if not cls.match(key, obj.metadata[key]):
            raise ValueError(f"Invalid metadata key for table result: {key}")

        # Parse the metadata entry as a TableResult dictionary
        table_dict = obj.metadata[key]
        table = TableResult.from_dict(table_dict)
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
