# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Adapter for Sigima's GeometryResult, providing features
for storing and retrieving those objects as metadata for DataLab's signal
and image objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generator

from sigima.objects import GeometryResult, ImageObj, SignalObj

if TYPE_CHECKING:
    import pandas as pd


class GeometryAdapter:
    """Adapter for GeometryResult objects.

    This adapter provides a unified interface for working with GeometryResult objects,
    including metadata storage/retrieval and various data representations.

    Args:
        geometry: GeometryResult object to adapt
    """

    # Class constants for metadata storage
    META_PREFIX: ClassVar[str] = "Geometry_"
    DICT_SUFFIX: ClassVar[str] = "_dict"

    def __init__(self, geometry: GeometryResult) -> None:
        self.geometry = geometry

    @classmethod
    def from_geometry_result(cls, geometry: GeometryResult) -> GeometryAdapter:
        """Create GeometryAdapter from GeometryResult.

        Args:
            geometry: GeometryResult object

        Returns:
            GeometryAdapter instance
        """
        return cls(geometry)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the geometry result to a pandas DataFrame.

        Returns:
            DataFrame with columns as in self.headers, and optional 'roi_index' column.
        """
        return self.geometry.to_dataframe()

    def get_roi_data(self, roi_index: int) -> pd.DataFrame:
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
        return [] if len(df) == 0 else [0]  # Default ROI index for geometry data

    @property
    def category(self) -> str:
        """Get the category.

        Returns:
            Category
        """
        return f"shape_{self.geometry.kind.value}"

    @property
    def title(self) -> str:
        """Get the title.

        Returns:
            Title
        """
        return self.geometry.title

    @property
    def headers(self) -> list[str]:
        """Get column headers for the coordinates.

        Returns:
            List of column headers
        """
        # Get headers directly from the DataFrame
        df = self.geometry.to_dataframe()
        # Return coordinate columns (exclude 'roi_index' if present)
        return [col for col in df.columns if col != "roi_index"]

    def add_to(self, obj: SignalObj | ImageObj) -> None:
        """Add geometry result to object metadata.

        Args:
            obj: Signal or image object
        """
        # Store the geometry as a single dictionary
        metadata_key = f"{self.META_PREFIX}{self.title}{self.DICT_SUFFIX}"
        obj.metadata[metadata_key] = self.geometry.to_dict()

    def remove_from(self, obj: SignalObj | ImageObj) -> None:
        """Remove geometry result from object metadata.

        Args:
            obj: Signal or image object
        """
        # Remove the single metadata key
        metadata_key = f"{self.META_PREFIX}{self.title}{self.DICT_SUFFIX}"
        obj.metadata.pop(metadata_key, None)

    @classmethod
    def remove_all_from(cls, obj: SignalObj | ImageObj) -> None:
        """Remove all geometry results from object metadata.

        Args:
            obj: Signal or image object
        """
        # Find all geometry results in the object and remove them
        for adapter in list(cls.iterate_from_obj(obj)):
            adapter.remove_from(obj)

    @property
    def label_contents(self) -> tuple[tuple[int, str], ...]:
        """Return label contents for compatibility.

        Returns:
            Tuple of couples of (index, text) where index is the column
            and text is the associated label
        """
        return tuple(enumerate(self.headers))

    def set_obj_metadata(self, obj: SignalObj | ImageObj) -> None:
        """Set object metadata from geometry result (alias for add_to).

        Args:
            obj: Signal or image object
        """
        self.add_to(obj)

    @staticmethod
    def add_geometry_result_to_obj(
        geometry: GeometryResult, obj: SignalObj | ImageObj
    ) -> None:
        """Static method to add GeometryResult to object.

        Args:
            geometry: GeometryResult object
            obj: Signal or image object
        """
        # Create adapter and add to object
        adapter = GeometryAdapter.from_geometry_result(geometry)
        adapter.add_to(obj)

    @classmethod
    def match(cls, key: str, _value: Any) -> bool:
        """Check if the key matches the pattern for a geometry result.

        Args:
            key: Metadata key
            _value: Metadata value (unused)

        Returns:
            True if the key matches the pattern
        """
        return key.startswith(cls.META_PREFIX) and key.endswith(cls.DICT_SUFFIX)

    @classmethod
    def from_metadata_entry(
        cls, obj: SignalObj | ImageObj, key: str
    ) -> GeometryAdapter:
        """Create a geometry result adapter from a metadata entry.

        Args:
            obj: Object containing the metadata
            key: Metadata key for the array data

        Returns:
            GeometryAdapter object

        Raises:
            ValueError: Invalid metadata entry
        """
        # Load the geometry data from the dictionary
        geometry_dict = obj.metadata[key]
        geometry = GeometryResult.from_dict(geometry_dict)
        return cls(geometry)

    @classmethod
    def iterate_from_obj(
        cls, obj: SignalObj | ImageObj
    ) -> Generator[GeometryAdapter, None, None]:
        """Iterate over geometry results stored in an object's metadata.

        Args:
            obj: Signal or image object

        Yields:
            GeometryAdapter objects
        """
        for key, value in obj.metadata.items():
            if cls.match(key, value):
                try:
                    yield cls.from_metadata_entry(obj, key)
                except (ValueError, TypeError):
                    # Skip invalid entries
                    pass
