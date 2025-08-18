"""
Adapter for Sigima's GeometryResult, providing features
for storing and retrieving those objects as metadata for DataLab's signal
and image objects.
"""

from __future__ import annotations

from typing import Any, ClassVar, Generator, Union

import numpy as np
from sigima.objects import NO_ROI, GeometryResult, ImageObj, KindShape, SignalObj


class GeometryAdapter:
    """Adapter for GeometryResult objects.

    This adapter provides a unified interface for working with GeometryResult objects,
    including metadata storage/retrieval and various data representations.

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

    def remove_from(self, obj: Union[SignalObj, ImageObj]) -> None:
        """Remove geometry result from object metadata.

        Args:
            obj: Signal or image object
        """
        base_key = f"{self.META_PREFIX}{self.title}"

        # Remove standard metadata keys
        keys_to_remove = [
            f"{base_key}{self.ARRAY_SUFFIX}",
            f"{base_key}{self.TITLE_SUFFIX}",
            f"{base_key}{self.SHAPE_SUFFIX}",
            f"{base_key}{self.ADDLABEL_SUFFIX}",
        ]

        # Remove any additional attribute keys
        if self.geometry.attrs:
            for key in self.geometry.attrs.keys():
                keys_to_remove.append(f"{base_key}_{key}")

        # Remove all keys that exist in the metadata
        for key in keys_to_remove:
            obj.metadata.pop(key, None)

    @classmethod
    def remove_all_from(cls, obj: Union[SignalObj, ImageObj]) -> None:
        """Remove all geometry results from object metadata.

        Args:
            obj: Signal or image object
        """
        # Find all geometry results in the object and remove them
        for adapter in cls.iterate_from_obj(obj):
            adapter.remove_from(obj)

    @property
    def label_contents(self) -> tuple[tuple[int, str], ...]:
        """Return label contents for compatibility.

        Returns:
            Tuple of couples of (index, text) where index is the column
            and text is the associated label
        """
        return tuple(enumerate(self.headers))

    def set_obj_metadata(self, obj: Union[SignalObj, ImageObj]) -> None:
        """Set object metadata from geometry result (alias for add_to).

        Args:
            obj: Signal or image object
        """
        self.add_to(obj)

    def get_text(self, obj: Union[SignalObj, ImageObj]) -> str:
        """Return text representation of result.

        Args:
            obj: Signal or image object for ROI title lookup

        Returns:
            HTML formatted text representation
        """
        text = ""
        for i_row in range(self.array.shape[0]):
            suffix = ""
            i_roi = i_row - 1
            if i_roi >= 0:
                suffix = f"|{obj.roi.get_single_roi_title(i_roi)}"
            text += f"<u>{self.title}{suffix}</u>:"
            for i_col, label in self.label_contents:
                label = label.replace("<", "&lt;").replace(">", "&gt;")
                if "%" not in label:
                    label += " = %g"
                value = self.shown_array[i_row, i_col]
                text += f"<br>{label.strip().format(obj) % value}"
            if i_row < self.shown_array.shape[0] - 1:
                text += "<br><br>"
        return text

    @staticmethod
    def add_geometry_result_to_obj(
        geometry: GeometryResult, obj: Union[SignalObj, ImageObj]
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

            # Convert shape string to KindShape
            if isinstance(shape_value, str):
                shape_map = {
                    "rectangle": KindShape.RECTANGLE,
                    "circle": KindShape.CIRCLE,
                    "ellipse": KindShape.ELLIPSE,
                    "segment": KindShape.SEGMENT,
                    "marker": KindShape.MARKER,
                    "point": KindShape.POINT,
                    "polygon": KindShape.POLYGON,
                }
                kind = shape_map.get(shape_value.lower(), KindShape.POINT)
            else:
                kind = shape_value

            geometry = GeometryResult(
                title=title,
                kind=kind,
                coords=coords,
                roi_indices=roi_indices,
                attrs={},
            )
        else:
            # Create empty GeometryResult
            if isinstance(shape_value, str):
                shape_map = {
                    "rectangle": KindShape.RECTANGLE,
                    "circle": KindShape.CIRCLE,
                    "ellipse": KindShape.ELLIPSE,
                    "segment": KindShape.SEGMENT,
                    "marker": KindShape.MARKER,
                    "point": KindShape.POINT,
                    "polygon": KindShape.POLYGON,
                }
                kind = shape_map.get(shape_value.lower(), KindShape.POINT)
            else:
                kind = shape_value

            geometry = GeometryResult(
                title=title,
                kind=kind,
                coords=np.zeros((0, 2), dtype=float),
                roi_indices=np.array([], dtype=int),
                attrs={},
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
