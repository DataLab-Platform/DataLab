# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Base Module
--------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from guidata.io import JSONReader, JSONWriter
from plotpy.io import load_items, save_items
from plotpy.items import (
    AbstractLabelItem,
    AnnotatedSegment,
    AnnotatedShape,
)

if TYPE_CHECKING:
    from plotpy.items import AbstractShape
    from plotpy.styles import AnnotationParam


def config_annotated_shape(
    item: AnnotatedShape,
    fmt: str,
    lbl: bool,
    section: str | None = None,
    option: str | None = None,
    show_computations: bool | None = None,
):
    """Configurate annotated shape

    Args:
        item: Annotated shape item
        fmt: Format string
        lbl: Show label
        section: Shape style section (e.g. "plot")
        option: Shape style option (e.g. "shape/drag")
        show_computations: Show computations
    """
    param: AnnotationParam = item.annotationparam
    param.format = fmt
    param.show_label = lbl
    if show_computations is not None:
        param.show_computations = show_computations

    if isinstance(item, AnnotatedSegment):
        item.label.labelparam.anchor = "T"
        item.label.labelparam.update_item(item.label)

    param.update_item(item)
    if section is not None and option is not None:
        item.set_style(section, option)


# TODO: [P3] Move this function as a method of plot items in PlotPy
def set_plot_item_editable(
    item: AbstractShape | AbstractLabelItem | AnnotatedShape, state
):
    """Set plot item editable state

    Args:
        item: Plot item
        state: State
    """
    item.set_movable(state)
    item.set_resizable(state and not isinstance(item, AbstractLabelItem))
    item.set_rotatable(state and not isinstance(item, AbstractLabelItem))
    item.set_readonly(not state)
    item.set_selectable(state)


def items_to_json(items: list) -> str | None:
    """Convert plot items to JSON string

    Args:
        items: list of plot items

    Returns:
        JSON string or None if items is empty
    """
    if items:
        writer = JSONWriter(None)
        save_items(writer, items)
        return writer.get_json(indent=4)
    return None


def json_to_items(json_str: str | None) -> list:
    """Convert JSON string to plot items

    Args:
        json_str: JSON string or None

    Returns:
        List of plot items
    """
    items = []
    if json_str:
        try:
            for item in load_items(JSONReader(json_str)):
                items.append(item)
        except json.decoder.JSONDecodeError:
            pass
    return items
