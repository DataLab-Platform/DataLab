# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Adapter for Sigima's TableResult, providing features
for storing and retrieving those objects as metadata for DataLab's signal
and image objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Union

from sigima.objects import ImageObj, SignalObj
from sigima.objects.scalar import NO_ROI, TableResult

from datalab.adapters_metadata.base_adapter import BaseResultAdapter

if TYPE_CHECKING:
    import pandas as pd


class TableAdapter(BaseResultAdapter):
    """Adapter for TableResult objects.

    This adapter provides a unified interface for working with TableResult objects,
    including metadata storage/retrieval and various data representations.

    Args:
        table: TableResult object to adapt
    """

    # Class constants for metadata storage
    META_PREFIX: ClassVar[str] = "Table_"
    SUFFIX: ClassVar[str] = "_data"

    def __init__(self, table: TableResult) -> None:
        super().__init__(table)
        self.table = table  # Keep for backwards compatibility

    @classmethod
    def from_table_result(cls, table: TableResult) -> TableAdapter:
        """Create TableAdapter from TableResult.

        Args:
            table: TableResult object

        Returns:
            TableAdapter instance
        """
        return cls(table)

    def to_dataframe(self) -> "pd.DataFrame":
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

    def get_unique_roi_indices(self) -> list[int]:
        """Get unique ROI indices present in the data.

        Returns:
            List of unique ROI indices
        """
        df = self.to_dataframe()
        if "roi_index" in df.columns:
            return sorted(df["roi_index"].unique().tolist())
        return [NO_ROI] if len(df) > 0 else []

    @classmethod
    def from_metadata_entry(cls, obj: Union[SignalObj, ImageObj], key: str):
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
