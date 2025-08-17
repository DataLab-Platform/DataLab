"""
Legacy Compatibility Layer
===========================

This module provides clean compatibility wrappers for old DataLab APIs,
replacing the need for monkey-patching Sigima classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sigima.objects.scalar import GeometryResult, TableResult

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

from sigima.objects.base import get_obj_roi_title

from .geometry_adapter import GeometryAdapter
from .table_adapter import TableAdapter


class LegacyGeometryResult:
    """Wrapper providing old ResultShape API for GeometryResult."""

    def __init__(self, geometry_result: GeometryResult):
        self._geometry_result = geometry_result
        self._adapter = GeometryAdapter.from_geometry_result(geometry_result)

    @property
    def title(self) -> str:
        return self._geometry_result.title

    @property
    def array(self) -> np.ndarray:
        return self._adapter.array

    @property
    def headers(self) -> list[str]:
        return self._adapter.headers

    @property
    def shown_array(self) -> np.ndarray:
        return self._adapter.shown_array

    @property
    def raw_data(self) -> np.ndarray:
        return self._adapter.raw_data

    @property
    def label_contents(self) -> tuple[tuple[int, str], ...]:
        return tuple(enumerate(self.headers))

    def add_to(self, obj: ImageObj | SignalObj) -> None:
        """Add geometry result to object."""
        self._adapter.add_to(obj)

    def set_obj_metadata(self, obj: ImageObj | SignalObj) -> None:
        """Set object metadata with geometry result."""
        self._adapter.add_to(obj)

    def get_text(self, obj: ImageObj | SignalObj) -> str:
        """Return text representation of result."""
        text = ""
        for i_row in range(self.array.shape[0]):
            suffix = ""
            i_roi = i_row - 1
            if i_roi >= 0:
                suffix = f"|{get_obj_roi_title(obj, i_roi)}"
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


class LegacyTableResult:
    """Wrapper providing old ResultProperties API for TableResult."""

    def __init__(self, table_result: TableResult):
        self._table_result = table_result
        self._adapter = TableAdapter.from_table_result(table_result)

    @property
    def title(self) -> str:
        return self._table_result.title

    @property
    def array(self) -> np.ndarray:
        return self._adapter.array

    @property
    def headers(self) -> list[str]:
        return self._adapter.headers

    @property
    def shown_array(self) -> np.ndarray:
        return self._adapter.shown_array

    @property
    def raw_data(self) -> np.ndarray:
        return self._adapter.raw_data

    @property
    def label_contents(self) -> tuple[tuple[int, str], ...]:
        return tuple(enumerate(self.headers))

    def add_to(self, obj: ImageObj | SignalObj) -> None:
        """Add table result to object."""
        self._adapter.add_to(obj)

    def set_obj_metadata(self, obj: ImageObj | SignalObj) -> None:
        """Set object metadata with table result."""
        self._adapter.add_to(obj)


def create_legacy_geometry_result(
    title: str, array: np.ndarray, shape: str, add_label: bool = False
) -> LegacyGeometryResult:
    """Create a legacy-compatible geometry result."""
    from sigima.objects.scalar import KindShape

    # Convert shape string to KindShape enum
    shape_map = {
        "rectangle": KindShape.RECTANGLE,
        "circle": KindShape.CIRCLE,
        "ellipse": KindShape.ELLIPSE,
        "segment": KindShape.SEGMENT,
        "marker": KindShape.MARKER,
        "point": KindShape.POINT,
        "polygon": KindShape.POLYGON,
    }

    kind = shape_map.get(shape.lower(), KindShape.RECTANGLE)

    # Extract coordinates and ROI indices from array
    coords = array[:, 1:]  # Skip first column (ROI indices)
    roi_indices = array[:, 0].astype(int) if array.shape[1] > 1 else np.array([0])

    geometry_result = create_geometry_result(
        title=title,
        kind=kind,
        coords=coords,
        roi_indices=roi_indices,
        attrs={"add_label": add_label},
    )

    return LegacyGeometryResult(geometry_result)


def create_legacy_table_result(
    title: str, array: np.ndarray, labels: list[str] | None = None
) -> LegacyTableResult:
    """Create a legacy-compatible table result."""

    # Extract data and ROI indices from array
    data = array[:, 1:]  # Skip first column (ROI indices)
    roi_indices = array[:, 0].astype(int) if array.shape[1] > 1 else np.array([0])

    table_result = create_table_result(
        title=title,
        data=data,
        headers=labels or [f"Col{i}" for i in range(data.shape[1])],
        roi_indices=roi_indices,
        attrs={},
    )

    return LegacyTableResult(table_result)


# Enhanced result classes that provide legacy interface without monkey-patching
class EnhancedGeometryResult(GeometryResult):
    """GeometryResult with legacy interface methods."""

    def add_to(self, obj):
        """Add geometry result to object metadata using GeometryAdapter"""
        adapter = GeometryAdapter.from_geometry_result(self)
        adapter.add_to(obj)

    @property
    def headers(self):
        """Return headers for geometry result using GeometryAdapter"""
        adapter = GeometryAdapter.from_geometry_result(self)
        return adapter.headers

    @property
    def array(self):
        """Return array for geometry result using GeometryAdapter"""
        adapter = GeometryAdapter.from_geometry_result(self)
        return adapter.array

    @property
    def shown_array(self):
        """Return shown array for geometry result using GeometryAdapter"""
        adapter = GeometryAdapter.from_geometry_result(self)
        return adapter.shown_array

    @property
    def raw_data(self):
        """Return raw data for geometry result using GeometryAdapter"""
        adapter = GeometryAdapter.from_geometry_result(self)
        return adapter.raw_data

    def to_dataframe(self):
        """Return DataFrame for geometry result using GeometryAdapter"""
        adapter = GeometryAdapter.from_geometry_result(self)
        return adapter.to_dataframe()

    @property
    def label_contents(self):
        """Return label contents, i.e. a tuple of couples of (index, text)
        where index is the column of raw_data and text is the associated
        label format string"""
        adapter = GeometryAdapter.from_geometry_result(self)
        headers = adapter.headers
        return tuple(enumerate(headers))

    def set_obj_metadata(self, obj):
        """Set object metadata from geometry result using GeometryAdapter"""
        adapter = GeometryAdapter.from_geometry_result(self)
        adapter.add_to(obj)


class EnhancedTableResult(TableResult):
    """TableResult with legacy interface methods."""

    def add_to(self, obj):
        """Add table result to object metadata using TableAdapter"""
        adapter = TableAdapter.from_table_result(self)
        adapter.add_to(obj)

    @property
    def headers(self):
        """Return headers for table result using TableAdapter"""
        adapter = TableAdapter.from_table_result(self)
        return adapter.headers

    @property
    def array(self):
        """Return array for table result using TableAdapter"""
        adapter = TableAdapter.from_table_result(self)
        return adapter.array

    @property
    def shown_array(self):
        """Return shown array for table result using TableAdapter"""
        adapter = TableAdapter.from_table_result(self)
        return adapter.shown_array

    @property
    def raw_data(self):
        """Return raw data for table result using TableAdapter"""
        adapter = TableAdapter.from_table_result(self)
        return adapter.raw_data

    def to_dataframe(self):
        """Return DataFrame for table result using TableAdapter"""
        adapter = TableAdapter.from_table_result(self)
        return adapter.to_dataframe()

    @property
    def label_contents(self):
        """Return label contents, i.e. a tuple of couples of (index, text)
        where index is the column and text is the associated label"""
        adapter = TableAdapter.from_table_result(self)
        headers = adapter.headers
        return tuple(enumerate(headers))

    def set_obj_metadata(self, obj):
        """Set object metadata from table result using TableAdapter"""
        adapter = TableAdapter.from_table_result(self)
        adapter.add_to(obj)


# Factory functions to create enhanced result objects
def create_geometry_result(title, kind, coords, roi_indices=None, attrs=None):
    """Create an enhanced GeometryResult with legacy interface."""
    # Ensure coords is a numpy array
    if not isinstance(coords, np.ndarray):
        coords = np.array(coords)

    if roi_indices is None:
        # Create roi_indices that match the number of coordinate rows
        roi_indices = np.arange(coords.shape[0], dtype=int)
    # Ensure roi_indices is a numpy array with integer dtype
    if not isinstance(roi_indices, np.ndarray):
        roi_indices = np.array(roi_indices, dtype=int)
    else:
        roi_indices = roi_indices.astype(int)

    return EnhancedGeometryResult(
        title=title,
        kind=kind,
        coords=coords,
        roi_indices=roi_indices,
        attrs=attrs or {},
    )


def create_table_result(
    title,
    names=None,
    labels=None,
    data=None,
    roi_indices=None,
    attrs=None,
    headers=None,
):
    """Create an enhanced TableResult with legacy interface.

    Args:
        title: Human-readable title for this table of results.
        names: Column names (one per metric) - for compatibility with Sigima.
        labels: Human-readable labels for each column.
        data: 2-D array of shape (N, len(names)) with scalar values.
        roi_indices: Optional 1-D array (N,) mapping rows to ROI indices.
                    If None, defaults to NO_ROI for all rows (no ROI association).
        attrs: Optional algorithmic context.
        headers: Alias for names for legacy compatibility.
    """
    # Use names or headers (headers is legacy compatibility)
    if names is None and headers is not None:
        names = headers
    elif names is None:
        names = []

    # Default empty data array if not provided
    if data is None:
        data = np.empty((0, len(names) if names else 0), float)

    # Ensure data is a numpy array
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if roi_indices is None:
        # Default to NO_ROI for all rows when no ROI indices are specified
        from sigima.objects.scalar import NO_ROI

        roi_indices = np.full(data.shape[0], NO_ROI, dtype=int)
    # Ensure roi_indices is a numpy array
    if not isinstance(roi_indices, np.ndarray):
        roi_indices = np.array(roi_indices)

    return EnhancedTableResult(
        title=title,
        names=names,
        labels=labels or [],
        data=data,
        roi_indices=roi_indices,
        attrs=attrs or {},
    )
