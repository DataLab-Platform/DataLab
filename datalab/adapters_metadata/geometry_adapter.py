# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Adapter for Sigima's GeometryResult, providing features
for storing and retrieving those objects as metadata for DataLab's signal
and image objects.
"""

from __future__ import annotations

from typing import ClassVar

from sigima.objects import GeometryResult, ImageObj, SignalObj

from datalab.adapters_metadata.base_adapter import BaseResultAdapter


class GeometryAdapter(BaseResultAdapter):
    """Adapter for GeometryResult objects.

    This adapter provides a unified interface for working with GeometryResult objects,
    including metadata storage/retrieval and various data representations.

    Args:
        geometry: GeometryResult object to adapt
    """

    # Class constants for metadata storage
    META_PREFIX: ClassVar[str] = "Geometry_"

    def __init__(self, geometry: GeometryResult) -> None:
        super().__init__(geometry)

    @classmethod
    def from_geometry_result(cls, geometry: GeometryResult) -> GeometryAdapter:
        """Create GeometryAdapter from GeometryResult.

        Args:
            geometry: GeometryResult object

        Returns:
            GeometryAdapter instance
        """
        return cls(geometry)

    @property
    def headers(self) -> list[str]:
        """Get column headers for the coordinates.

        Returns:
            List of column headers
        """
        # Get headers directly from the DataFrame
        df = self.result.to_dataframe()
        # Return coordinate columns (exclude 'roi_index' if present)
        return [col for col in df.columns if col != "roi_index"]

    @property
    def category(self) -> str:
        """Get the category.

        Returns:
            Category
        """
        return f"shape_{self.result.kind.value}"

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
    def from_metadata_entry(cls, obj: SignalObj | ImageObj, key: str):
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
