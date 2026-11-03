# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Annotation tools unit test:

  - For each annotation tool offered by the signal and image panels, draw a shape in
    the annotation editor and validate it: the shape must be stored as a portable
    annotation of the expected type, and validating the editor again must leave the
    annotations unchanged (see Issue #476)
  - Rectangles and segments stored as plain PlotPy shapes by DataLab 1.3 or earlier
    must keep the changes made in the annotation editor
  - Segments drawn on signals must not show computations mixing X and Y units
"""

# guitest: show

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import guidata.dataset as gds
import numpy as np
import pytest
from guidata.io import JSONWriter
from plotpy.builder import make
from plotpy.io import save_items
from plotpy.items import AnnotatedSegment, RectangleShape, SegmentShape
from plotpy.tests.unit.utils import drag_mouse
from plotpy.tools import (
    AnnotatedCircleTool,
    AnnotatedEllipseTool,
    AnnotatedPointTool,
    AnnotatedRectangleTool,
    AnnotatedSegmentTool,
    HCursorTool,
    HRangeTool,
    LabelTool,
    VCursorTool,
    XCursorTool,
)
from qtpy import QtWidgets as QW
from sigima.objects import (
    CircleAnnotation,
    CursorAnnotation,
    EllipseAnnotation,
    PointAnnotation,
    RangeAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    TextAnnotation,
)
from sigima.tests import data as test_data

from datalab.env import execenv
from datalab.gui.panel.image import ImagePanel
from datalab.gui.panel.signal import SignalAnnotatedSegmentTool, SignalPanel
from datalab.tests import datalab_test_app_context

if TYPE_CHECKING:
    from collections.abc import Callable

    from plotpy.plot import PlotDialog
    from plotpy.tools import InteractiveTool
    from sigima.objects import ImageObj, SignalObj

    from datalab.gui.panel.base import BaseDataPanel

#: Type of the portable annotation created by each annotation tool
EXPECTED_ANNOTATION_TYPES = {
    LabelTool: TextAnnotation,
    VCursorTool: CursorAnnotation,
    HCursorTool: CursorAnnotation,
    XCursorTool: CursorAnnotation,
    HRangeTool: RangeAnnotation,
    SignalAnnotatedSegmentTool: SegmentAnnotation,
    AnnotatedSegmentTool: SegmentAnnotation,
    AnnotatedRectangleTool: RectangleAnnotation,
    AnnotatedCircleTool: CircleAnnotation,
    AnnotatedEllipseTool: EllipseAnnotation,
    AnnotatedPointTool: PointAnnotation,
}


#: Text entered in the dialog opened by the label tool
LABEL_TEXT = "Label text"


def test_annotation_tools_are_all_tested() -> None:
    """Every annotation tool offered by a panel must be tested below"""
    for panel_class in (SignalPanel, ImagePanel):
        untested = set(panel_class.ANNOTATION_TOOLS) - set(EXPECTED_ANNOTATION_TYPES)
        assert not untested, f"{panel_class.__name__}: {untested}"


def enter_label_text(dataset: gds.DataSet, *_args, **_kwargs) -> bool:
    """Enter the label text instead of opening the label tool dialog

    The modal dialog is not used, because its closing in unattended mode depends on
    the timers left by previous tests.
    """
    dataset.text = LABEL_TEXT
    return True


def draw_with_tool(dlg: PlotDialog, tool_class: type[InteractiveTool]) -> list:
    """Draw a shape in the annotation editor with an annotation tool

    Args:
        dlg: Annotation editor
        tool_class: Annotation tool class

    Returns:
        Plot items created by the tool
    """
    plot = dlg.get_plot()
    items_before = set(plot.get_items())
    tool = dlg.get_manager().get_tool(tool_class)
    assert tool is not None, f"{tool_class.__name__} not found"
    tool.activate()
    with patch.object(gds.DataSet, "edit", enter_label_text):
        drag_mouse(dlg, np.linspace(0.3, 0.6, 10), np.linspace(0.3, 0.6, 10))
    QW.QApplication.processEvents()
    return [item for item in plot.get_items() if item not in items_before]


def open_annotation_editor(panel: BaseDataPanel) -> PlotDialog:
    """Open the annotation editor of the current object

    Args:
        panel: Signal or image panel

    Returns:
        Annotation editor
    """
    dlg = panel.open_separate_view(edit_annotations=True)
    assert dlg is not None
    QW.QApplication.processEvents()
    return dlg


def check_annotation_tools(
    panel: BaseDataPanel, create_object: Callable[[], SignalObj | ImageObj]
) -> None:
    """Check that every annotation tool of a panel creates a stored annotation

    Args:
        panel: Signal or image panel
        create_object: Function creating a new object to annotate
    """
    for tool_class in panel.ANNOTATION_TOOLS:
        execenv.print(f"  {tool_class.__name__}: ", end="")
        obj = create_object()
        obj.title = tool_class.__name__
        panel.add_object(obj)

        dlg = open_annotation_editor(panel)
        new_items = draw_with_tool(dlg, tool_class)
        assert len(new_items) == 1, f"{tool_class.__name__} created {new_items}"
        dlg.done(QW.QDialog.DialogCode.Accepted)

        annotations = obj.get_annotations()
        assert len(annotations) == 1, f"{tool_class.__name__}: shape was not kept"
        assert annotations[0]["format"] == "sigima.annotation"
        [annotation] = obj.get_graphical_annotations()
        assert isinstance(annotation, EXPECTED_ANNOTATION_TYPES[tool_class])
        if isinstance(annotation, TextAnnotation):
            assert annotation.text == LABEL_TEXT

        open_annotation_editor(panel).done(QW.QDialog.DialogCode.Accepted)
        assert obj.get_annotations() == annotations
        execenv.print(type(annotation).__name__)


def test_annotation_tools() -> None:
    """Each annotation tool must create an annotation kept by the editor"""
    with datalab_test_app_context() as win:
        execenv.print("Signal annotation tools:")
        check_annotation_tools(win.signalpanel, test_data.create_paracetamol_signal)
        execenv.print("Image annotation tools:")
        check_annotation_tools(win.imagepanel, test_data.create_multigaussian_image)


def test_signal_segment_without_computations() -> None:
    """Segments drawn on signals must not show their length and angle"""
    with datalab_test_app_context() as win:
        panel = win.signalpanel
        panel.add_object(test_data.create_paracetamol_signal())
        dlg = open_annotation_editor(panel)
        [segment] = draw_with_tool(dlg, SignalAnnotatedSegmentTool)
        assert isinstance(segment, AnnotatedSegment)
        assert not segment.area_computations_visible
        assert segment.get_info() not in segment.get_text()
        dlg.done(QW.QDialog.DialogCode.Rejected)


def make_legacy_payload(item: RectangleShape | SegmentShape) -> dict:
    """Return the annotation payload stored by DataLab 1.3 for a PlotPy item"""
    writer = JSONWriter(None)
    save_items(writer, [item])
    return {
        "type": "plotpy_item",
        "item_class": type(item).__name__,
        "plotpy_json": writer.get_json(),
    }


def test_edit_legacy_signal_rectangle_and_segment() -> None:
    """Changes of plain rectangles and segments stored by DataLab 1.3 must be kept"""
    with datalab_test_app_context() as win:
        panel = win.signalpanel
        sig = test_data.create_paracetamol_signal()
        sig.set_annotations(
            [
                make_legacy_payload(make.rectangle(10.0, 0.1, 20.0, 0.5, "Rect")),
                make_legacy_payload(make.segment(10.0, 0.1, 20.0, 0.5, "Seg")),
            ]
        )
        panel.add_object(sig)

        dlg = open_annotation_editor(panel)
        shapes = [
            item
            for item in dlg.get_plot().get_items()
            if isinstance(item, (RectangleShape, SegmentShape))
        ]
        assert len(shapes) == 2
        for shape in shapes:
            assert not shape.is_readonly()
            shape.move_with_selection(5.0, 0.2)
        dlg.done(QW.QDialog.DialogCode.Accepted)

        assert all(
            payload["format"] == "sigima.annotation"
            for payload in sig.get_annotations()
        )
        rectangle, segment = sig.get_graphical_annotations()
        assert isinstance(rectangle, RectangleAnnotation)
        assert rectangle.title == "Rect"
        assert (rectangle.x, rectangle.y) == pytest.approx((20.0, 0.5))
        assert (rectangle.width, rectangle.height) == pytest.approx((10.0, 0.4))
        assert isinstance(segment, SegmentAnnotation)
        assert segment.title == "Seg"
        assert (segment.x0, segment.y0, segment.x1, segment.y1) == pytest.approx(
            (15.0, 0.3, 25.0, 0.7)
        )


if __name__ == "__main__":
    test_annotation_tools_are_all_tested()
    test_annotation_tools()
    test_signal_segment_without_computations()
    test_edit_legacy_signal_rectangle_and_segment()
