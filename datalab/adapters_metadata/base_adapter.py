# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Base adapter class for result objects, providing common functionality
for storing and retrieving result objects as metadata for DataLab's signal
and image objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generator

from sigima.objects import ImageObj, SignalObj

if TYPE_CHECKING:
    import pandas as pd
    from sigima.objects.scalar import GeometryResult, TableResult


class BaseResultAdapter(ABC):
    """Base adapter for result objects.

    This base class provides common functionality for working with result objects,
    including metadata storage/retrieval and various data representations.

    Args:
        result: Result object to adapt
    """

    # Class constants for metadata storage - to be overridden by subclasses
    META_PREFIX: ClassVar[str] = ""
    SUFFIX: ClassVar[str] = ""

    def __init__(self, result: TableResult | GeometryResult) -> None:
        self.result = result

    def set_applicative_attr(self, key: str, value: Any) -> None:
        """Set an applicative attribute for the result.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.result.attrs[key] = value

    def get_applicative_attr(self, key: str, default: Any = None) -> Any:
        """Get an applicative attribute for the result.

        Args:
            key: Attribute key
            default: Default value to return if key not found

        Returns:
            Attribute value, or default if not set. If default is not None, assign it
            to the attribute if it was not already set.
        """
        if key not in self.result.attrs and default is not None:
            self.result.attrs[key] = default
        return self.result.attrs.get(key, default)

    @property
    def title(self) -> str:
        """Get the title.

        Returns:
            Title
        """
        return self.result.title

    @property
    @abstractmethod
    def headers(self) -> list[str]:
        """Get column headers.

        Returns:
            List of column headers
        """

    @property
    @abstractmethod
    def category(self) -> str:
        """Get the category.

        Returns:
            Category
        """

    def get_roi_data(self, roi_index: int) -> "pd.DataFrame":
        """Get data for a specific ROI index.

        Args:
            roi_index: ROI index to filter by

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
        return [] if len(df) == 0 else [0]  # Default ROI index for result data

    def add_to(self, obj: SignalObj | ImageObj) -> None:
        """Add result to object metadata.

        Args:
            obj: Signal or image object
        """
        # Store the result as a single dictionary
        metadata_key = f"{self.META_PREFIX}{self.title}{self.SUFFIX}"
        obj.metadata[metadata_key] = self.result.to_dict()

    def remove_from(self, obj: SignalObj | ImageObj) -> None:
        """Remove result from object metadata.

        Args:
            obj: Signal or image object
        """
        # Remove the single metadata key
        metadata_key = f"{self.META_PREFIX}{self.title}{self.SUFFIX}"
        obj.metadata.pop(metadata_key, None)

    @classmethod
    def remove_all_from(cls, obj: SignalObj | ImageObj) -> None:
        """Remove all results of this type from object metadata.

        Args:
            obj: Signal or image object
        """
        # Find all results in the object and remove them
        for adapter in list(cls.iterate_from_obj(obj)):
            adapter.remove_from(obj)

    @classmethod
    def match(cls, key: str, _value: Any) -> bool:
        """Check if the key matches the pattern for this result type.

        Args:
            key: Metadata key
            _value: Metadata value (unused)

        Returns:
            True if the key matches the pattern
        """
        return key.startswith(cls.META_PREFIX) and key.endswith(cls.SUFFIX)

    @classmethod
    @abstractmethod
    def from_metadata_entry(cls, obj: SignalObj | ImageObj, key: str):
        """Create a result adapter from a metadata entry.

        Args:
            obj: Object containing the metadata
            key: Metadata key for the result data

        Returns:
            Adapter object

        Raises:
            ValueError: Invalid metadata entry
        """

    @classmethod
    def iterate_from_obj(
        cls, obj: SignalObj | ImageObj
    ) -> Generator["BaseResultAdapter", None, None]:
        """Iterate over results stored in an object's metadata.

        Args:
            obj: Signal or image object

        Yields:
            Adapter objects
        """
        for key, value in obj.metadata.items():
            if cls.match(key, value):
                try:
                    yield cls.from_metadata_entry(obj, key)
                except (ValueError, TypeError):
                    # Skip invalid entries
                    pass

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert the geometry result to a pandas DataFrame.

        Returns:
            DataFrame with columns as in self.headers, and optional 'roi_index' column.
        """
        return self.result.to_dataframe()

    def to_html(
        self,
        obj=None,
        visible_headers: list[str] = None,
        transpose_single_row: bool = True,
        **kwargs,
    ) -> str:
        """Convert the result to HTML format.

        Args:
            obj: Optional SignalObj or ImageObj for ROI title extraction
            visible_headers: Optional list of headers to show (filters columns)
            transpose_single_row: If True, transpose the table when there's only one row
            **kwargs: Additional arguments passed to DataFrame.to_html()

        Returns:
            HTML representation of the result
        """
        # Use visible headers from display preferences if not specified
        if visible_headers is None:
            visible_headers = self.result.get_visible_headers()

        return self.result.to_html(
            obj=obj,
            visible_headers=visible_headers,
            transpose_single_row=transpose_single_row,
            **kwargs,
        )

    def get_display_preferences(self) -> dict[str, bool]:
        """Get display preferences.

        Returns:
            Dictionary mapping header names to visibility (True=visible, False=hidden)
        """
        return self.result.get_display_preferences()

    def set_display_preferences(self, preferences: dict[str, bool]) -> None:
        """Set display preferences.

        Args:
            preferences: Dictionary mapping header names to visibility
        """
        self.result.set_display_preferences(preferences)

    def get_visible_headers(self) -> list[str]:
        """Get list of currently visible headers based on display preferences.

        Returns:
            List of header names that should be displayed
        """
        return self.result.get_visible_headers()
