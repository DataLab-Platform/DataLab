# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Base adapter class for result objects, providing common functionality
for storing and retrieving result objects as metadata for DataLab's signal
and image objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Generator

import guidata.dataset as gds
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
    META_SUFFIX: ClassVar[str] = "_dict"

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
    def name(self) -> str:
        """Get the result kind name.

        Returns:
            The string value of the kind attribute (e.g., "segment", "circle",
            "statistics"). This is NOT unique - multiple results can share the
            same kind.
        """
        return self.result.name

    @property
    def func_name(self) -> str:
        """Get the computation function name.

        Returns:
            The name of the computation function that produced this result.
        """
        return self.result.func_name

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

    @property
    def metadata_key(self) -> str:
        """Get the metadata key used to store this result.

        Returns:
            Metadata key based on the result's title
        """
        return f"{self.META_PREFIX}{self.func_name}{self.META_SUFFIX}"

    def add_to(
        self, obj: SignalObj | ImageObj, param: gds.DataSet | None = None
    ) -> None:
        """Add result to object metadata.

        Args:
            obj: Signal or image object
            param: Optional parameter dataset associated with this result
        """
        assert self.func_name, "func_name must be set before adding to object metadata"
        # Store parameter in result attrs (will be serialized with result)
        if param is not None:
            self.result.attrs["param_json"] = gds.dataset_to_json(param)
        obj.metadata[self.metadata_key] = self.result.to_dict()

    def get_param(self) -> gds.DataSet | None:
        """Get parameter dataset associated with this result.

        Returns:
            Parameter dataset if present, None otherwise
        """
        param_json = self.result.attrs.get("param_json")
        if param_json is not None:
            return gds.json_to_dataset(param_json)
        return None

    def remove_from(self, obj: SignalObj | ImageObj) -> None:
        """Remove result from object metadata.

        Args:
            obj: Signal or image object
        """
        obj.metadata.pop(self.metadata_key, None)

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
        return key.startswith(cls.META_PREFIX) and key.endswith(cls.META_SUFFIX)

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
    def from_obj(
        cls, obj: SignalObj | ImageObj, func_name: str
    ) -> BaseResultAdapter | None:
        """Create a result adapter from an object's metadata entry.

        Args:
            obj: Signal or image object
            func_name: Function name to look for

        Returns:
            Adapter object if found, None otherwise
        """
        for adapter in cls.iterate_from_obj(obj):
            if adapter.func_name == func_name:
                return adapter
        return None

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

    def to_dataframe(self, visible_only: bool = False):
        """Convert the result to a pandas DataFrame.

        Args:
            visible_only: If True, include only visible headers based on display
             preferences. Default is False.

        Returns:
            DataFrame with an optional 'roi_index' column.
             If visible_only is True, only columns with visible headers are included.
        """
        return self.result.to_dataframe(visible_only=visible_only)

    def to_html(
        self,
        obj=None,
        visible_only: bool = True,
        transpose_single_row: bool = True,
        **kwargs,
    ) -> str:
        """Convert the result to HTML format.

        Args:
            obj: Optional SignalObj or ImageObj for ROI title extraction
            visible_only: If True, include only visible headers based on display
             preferences. Default is False.
            transpose_single_row: If True, transpose the table when there's only one row
            **kwargs: Additional arguments passed to DataFrame.to_html()

        Returns:
            HTML representation of the result
        """
        return self.result.to_html(
            obj=obj,
            visible_only=visible_only,
            transpose_single_row=transpose_single_row,
            **kwargs,
        )

    def get_visible_headers(self) -> list[str]:
        """Get list of currently visible headers based on display preferences.

        Returns:
            List of header names that should be displayed
        """
        return self.result.get_visible_headers()
