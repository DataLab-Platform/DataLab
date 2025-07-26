# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Annotations unit test:

  - Create an image with annotations
  - Open dialog (equivalent to click on button "Annotations")
  - Accept dialog without modifying shapes
  - Check if image annotations are still the same
"""

# guitest: show

from plotpy.builder import make
from plotpy.items import AnnotatedShape, PolygonShape
from plotpy.plot import BasePlot
from qtpy import QtWidgets as QW
from sigima.tests import data as test_data

from datalab.adapters_plotpy.factories import create_adapter_from_object
from datalab.env import execenv
from datalab.tests import cdltest_app_context


def set_annotation_color(annotation: AnnotatedShape, color: str) -> None:
    """Set annotation color"""
    shape: PolygonShape = annotation.shape
    param = shape.shapeparam
    param.line.color = param.fill.color = color
    param.fill.alpha = 0.3
    param.fill.style = "SolidPattern"
    param.update_item(shape)
    plot: BasePlot = annotation.plot()
    if plot is not None:
        plot.replot()


def test_annotations_unit():
    """Run image tools test scenario"""
    with cdltest_app_context() as win:
        panel = win.imagepanel

        # Create image with annotations
        ima1 = test_data.create_multigaussian_image()
        ima1.title = "Annotations from items"
        rect = make.annotated_rectangle(100, 100, 200, 200, title="Test")
        set_annotation_color(rect, "#2222ff")
        circ = make.annotated_circle(300, 300, 400, 400, title="Test")
        set_annotation_color(circ, "#22ff22")
        elli = make.annotated_ellipse(
            500, 500, 800, 500, 650, 400, 650, 600, title="Test"
        )
        segm = make.annotated_segment(700, 700, 800, 800, title="Test")
        label = make.label("Test", (1000, 1000), (0, 0), "BR")
        adapter = create_adapter_from_object(ima1)
        adapter.add_annotations_from_items([rect, circ, elli, segm, label])
        panel.add_object(ima1)

        # Create another image with annotations
        ima2 = test_data.create_annotated_image(title="Annotations from JSON")
        panel.add_object(ima2)

        execenv.print("Check [geometric shapes] <--> [plot items] conversion:")
        execenv.print("(comparing annotations)")
        for ima in (ima1, ima2):
            execenv.print(f"  Checking image '{ima.title}': ", end="")
            panel.objview.select_objects([ima])
            # Open separate view
            panel.open_separate_view().done(QW.QDialog.DialogCode.Accepted)
            orig_ann = ima.annotations
            panel.open_separate_view().done(QW.QDialog.DialogCode.Accepted)
            # Check if annotations are still the same
            # open("before.json", mode="wb").write(orig_ann.encode())
            # open("after.json", mode="wb").write(ima.annotations.encode())
            assert orig_ann == ima.annotations
            execenv.print("OK")


if __name__ == "__main__":
    test_annotations_unit()
