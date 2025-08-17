"""
Adapter for Sigima's GeometryResult, providing features
for storing and retrieving those objects as metadata for DataLab's signal
and image objects.
"""

from __future__ import annotations

from typing import (
    Any,
    ClassVar,
    Generator,
    Union,
)

import numpy as np
from sigima.objects import ImageObj, SignalObj
from sigima.objects.scalar import NO_ROI, GeometryResult, KindShape


class GeometryAdapter:
    """Adapter for GeometryResult objects.

    This adapter provides compatibility with the old ResultShape interface
    for DataLab, particularly for metadata storage and retrieval.

    Args:
        geometry: GeometryResult object to adapt
    """

    # Class constants for metadata storage
    META_PREFIX: ClassVar[str] = "Geometry_"
    ARRAY_SUFFIX: ClassVar[str] = "_array"
    TITLE_SUFFIX: ClassVar[str] = "_title"
    SHAPE_SUFFIX: ClassVar[str] = "_shape"
    ADDLABEL_SUFFIX: ClassVar[str] = "_addlabel"

    def __init__(self, geometry: GeometryResult) -> None:
        self.geometry = geometry
        # Convert GeometryResult data to the format expected by DataLab
        self._array = self._prepare_array()

    @classmethod
    def from_geometry_result(cls, geometry: GeometryResult) -> GeometryAdapter:
        """Create GeometryAdapter from GeometryResult.

        Args:
            geometry: GeometryResult object

        Returns:
            GeometryAdapter instance
        """
        return cls(geometry)

    def _prepare_array(self) -> np.ndarray:
        """Convert GeometryResult coordinates to the format expected by DataLab.

        Returns:
            Array with ROI indices in the first column and coordinates in the following
            columns
        """
        # Create array with ROI indices as the first column
        rows = self.geometry.coords.shape[0]
        cols = self.geometry.coords.shape[1] + 1  # +1 for ROI indices column
        result = np.zeros((rows, cols), dtype=float)

        # Set ROI indices as integers (convert to int when setting to preserve dtype)
        if self.geometry.roi_indices is not None:
            result[:, 0] = self.geometry.roi_indices.astype(int)
        else:
            result[:, 0] = NO_ROI

        # Set coordinates
        result[:, 1:] = self.geometry.coords

        return result

    @property
    def shapetype(self) -> KindShape:
        """Get the shape type.

        Returns:
            Shape type as KindShape enum
        """
        return self.geometry.kind

    @property
    def category(self) -> str:
        """Get the category.

        Returns:
            Category
        """
        return "shape"

    @property
    def title(self) -> str:
        """Get the title.

        Returns:
            Title
        """
        return self.geometry.title

    @property
    def array(self) -> np.ndarray:
        """Get the array with ROI indices and coordinates.

        Returns:
            Array with ROI indices in first column and coordinates in the following
            columns
        """
        return self._array

    @property
    def raw_data(self) -> np.ndarray:
        """Get raw data (coordinates without ROI indices).

        Returns:
            Array of coordinates (without ROI indices)
        """
        return self._array[:, 1:]

    @property
    def shown_array(self) -> np.ndarray:
        """Get the shown array (same as raw_data for basic geometry).

        Returns:
            Array shown to the user
        """
        return self.raw_data

    @property
    def headers(self) -> list[str]:
        """Get column headers for the coordinates.

        Returns:
            List of column headers
        """
        # Create headers based on the shape type
        kind = self.geometry.kind.value
        num_coords = self._array.shape[1] - 1  # Exclude ROI column

        # Define headers based on shape type
        headers_map = {
            "point": ["x", "y"],
            "marker": ["x", "y"],
            "segment": ["x0", "y0", "x1", "y1"],
            "rectangle": ["x0", "y0", "x1", "y1"],
            "circle": ["x", "y", "r"],
            "ellipse": ["x", "y", "a", "b", "θ"],
        }

        if kind in headers_map:
            return headers_map[kind][:num_coords]

        if kind == "polygon":
            headers = []
            for i in range(0, num_coords, 2):
                headers.extend([f"x{i // 2}", f"y{i // 2}"])
            return headers[:num_coords]

        # Generic headers for unknown shapes
        return [f"coord_{i}" for i in range(num_coords)]

    def to_dataframe(self):
        """Return DataFrame from coordinates array.

        Returns:
            pandas.DataFrame with coordinates data
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        return pd.DataFrame(self.shown_array, columns=self.headers)

    def add_to(self, obj: Union[SignalObj, ImageObj]) -> None:
        """Add geometry result to object metadata.

        Args:
            obj: Signal or image object
        """
        # Create a unique key for this shape
        base_key = f"{self.META_PREFIX}{self.title}"

        # Store array
        obj.metadata[f"{base_key}{self.ARRAY_SUFFIX}"] = self._array.tolist()

        # Store title
        obj.metadata[f"{base_key}{self.TITLE_SUFFIX}"] = self.title

        # Store shape type
        obj.metadata[f"{base_key}{self.SHAPE_SUFFIX}"] = self.geometry.kind.value

        # Store add_label flag (defaults to False for backward compatibility)
        obj.metadata[f"{base_key}{self.ADDLABEL_SUFFIX}"] = False

        # Store any additional attributes from the GeometryResult
        if self.geometry.attrs:
            for key, value in self.geometry.attrs.items():
                obj.metadata[f"{base_key}_{key}"] = value

    @staticmethod
    def add_geometry_result_to_obj(
        geometry: GeometryResult, obj: Union[SignalObj, ImageObj]
    ) -> None:
        """Static method to add GeometryResult to object, automatically converting
        to enhanced version if needed.

        Args:
            geometry: GeometryResult object (raw or enhanced)
            obj: Signal or image object
        """
        # Check if this is already an enhanced GeometryResult
        from .legacy import EnhancedGeometryResult

        if not isinstance(geometry, EnhancedGeometryResult):
            # Convert raw GeometryResult to enhanced version
            from .legacy import create_geometry_result

            geometry = create_geometry_result(
                title=geometry.title,
                kind=geometry.kind,
                coords=geometry.coords,
                roi_indices=geometry.roi_indices,
                attrs=geometry.attrs,
            )

        # Use the enhanced geometry's add_to method
        geometry.add_to(obj)

    @classmethod
    def match(cls, key: str, _value: Any) -> bool:
        """Check if the key matches the pattern for a geometry result.

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
        if not cls.match(key, obj.metadata[key]):
            raise ValueError(f"Invalid metadata key for geometry result: {key}")

        base_key = key[: -len(cls.ARRAY_SUFFIX)]
        title = base_key[len(cls.META_PREFIX) :]

        # Parse the metadata entry
        array_data = obj.metadata[key]
        array = np.array(array_data, dtype=float)

        # Get shape type
        shape_key = f"{base_key}{cls.SHAPE_SUFFIX}"
        shape_value = "point"  # Default for backward compatibility
        if shape_key in obj.metadata:
            shape_value = obj.metadata[shape_key]

        if array.size > 0:
            # Create GeometryResult from the data
            roi_indices = array[:, 0].astype(int)
            coords = array[:, 1:]

            from datalab.adapters_metadata.legacy import create_geometry_result

            geometry = create_geometry_result(
                title=title,
                kind=shape_value,
                coords=coords,
                roi_indices=roi_indices,
            )
        else:
            # Create empty GeometryResult
            geometry = create_geometry_result(
                title=title,
                kind=KindShape(shape_value),
                coords=np.zeros((0, 2), dtype=float),
            )
        return cls(geometry)

    @classmethod
    def iterate_from_obj(
        cls, obj: Union[SignalObj, ImageObj]
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
