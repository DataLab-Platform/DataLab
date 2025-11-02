# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Annotation Adapter for PlotPy Integration
-----------------------------------------

This module bridges Sigima's format-agnostic annotation storage with PlotPy's
plot item system. It handles bidirectional conversion between:
- Sigima: list[dict] (JSON-serializable)
- PlotPy: list[AnnotatedShape] (plot items)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from guidata.io import JSONReader, JSONWriter
from plotpy.io import load_items, save_items

if TYPE_CHECKING:
    from plotpy.items import AnnotatedShape
    from sigima.objects.base import BaseObj


class PlotPyAnnotationAdapter:
    """Adapter for converting between Sigima annotations and PlotPy items.

    This class provides the bridge between Sigima's generic annotation storage
    (list of dicts) and PlotPy's specific plot item format.

    Example:
        >>> from sigima.objects.signal.creation import create_signal
        >>> obj = create_signal("Test")
        >>> adapter = PlotPyAnnotationAdapter(obj)
        >>>
        >>> # Add PlotPy items
        >>> from plotpy.items import AnnotatedRectangle
        >>> rect = AnnotatedRectangle(0, 0, 10, 10)
        >>> adapter.add_items([rect])
        >>>
        >>> # Retrieve as PlotPy items
        >>> items = adapter.get_items()
        >>> len(items)
        1
    """

    def __init__(self, obj: BaseObj):
        """Initialize adapter with an object.

        Args:
            obj: Signal or image object with annotation support
        """
        self.obj = obj

    def get_items(self) -> list[AnnotatedShape]:
        """Get annotations as PlotPy items.

        Returns:
            List of PlotPy annotation items

        Notes:
            This method deserializes the JSON data stored in the object using
            PlotPy's load_items() function.
        """
        annotations = self.obj.get_annotations()
        if not annotations:
            return []

        items = []
        for ann_dict in annotations:
            # Each annotation dict should contain PlotPy's JSON serialization
            if "plotpy_json" in ann_dict:
                try:
                    json_str = ann_dict["plotpy_json"]
                    for item in load_items(JSONReader(json_str)):
                        items.append(item)
                except (json.JSONDecodeError, ValueError, KeyError):
                    # Skip invalid items
                    continue

        return items

    def set_items(self, items: list[AnnotatedShape]) -> None:
        """Set annotations from PlotPy items.

        Args:
            items: List of PlotPy annotation items

        Notes:
            This method serializes PlotPy items to JSON using PlotPy's
            save_items() function and stores them in the Sigima format.
        """
        if not items:
            self.obj.clear_annotations()
            return

        # Convert PlotPy items to our annotation format
        annotations = []
        for item in items:
            writer = JSONWriter(None)
            save_items(writer, [item])
            ann_dict = {
                "type": "plotpy_item",
                "item_class": item.__class__.__name__,
                "plotpy_json": writer.get_json(),
            }
            annotations.append(ann_dict)

        self.obj.set_annotations(annotations)

    def add_items(self, items: list[AnnotatedShape]) -> None:
        """Add PlotPy items to existing annotations.

        Args:
            items: List of PlotPy annotation items to add
        """
        current_items = self.get_items()
        current_items.extend(items)
        self.set_items(current_items)

    def clear(self) -> None:
        """Clear all annotations."""
        self.obj.clear_annotations()
