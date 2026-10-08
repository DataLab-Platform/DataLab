# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Annotations unit test:

  - Create an image with annotations
  - Open dialog (equivalent to click on button "Annotations")
  - Accept dialog without modifying shapes
  - Check if image annotations are still the same
"""

# guitest: show

from guidata.io import JSONWriter
from plotpy.builder import make
from plotpy.io import save_items
from plotpy.items import AnnotatedShape, PolygonShape
from plotpy.plot import BasePlot
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import RectangleAnnotation, annotation_to_dict, create_image_roi
from sigima.tests import data as test_data

from datalab.adapters_plotpy import create_adapter_from_object
from datalab.env import execenv
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


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
    with datalab_test_app_context() as win:
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
        assert all(
            payload["format"] == "sigima.annotation"
            for payload in ima1.get_annotations()
        )
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


def test_separate_view_migrates_annotations_only_when_accepted() -> None:
    """The annotation dialog migrates lazily and preserves opaque state."""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        image = test_data.create_multigaussian_image()
        canonical = annotation_to_dict(
            RectangleAnnotation(
                x=3.0,
                y=5.0,
                width=4.0,
                height=6.0,
                locked=True,
                title="Locked",
                metadata={"owner": "test"},
                extensions={"vendor": {"keep": True}},
            )
        )
        legacy_item = make.annotated_segment(1.0, 2.0, 5.0, 8.0, title="Legacy")
        writer = JSONWriter(None)
        save_items(writer, [legacy_item])
        legacy = {
            "type": "plotpy_item",
            "item_class": type(legacy_item).__name__,
            "plotpy_json": writer.get_json(),
        }
        opaque = {"consumer": "custom", "payload": {"keep": True}}
        image.set_annotations([canonical, legacy, opaque])
        image.roi = create_image_roi("rectangle", [10, 20, 30, 40])
        original_roi = image.roi.to_dict()
        panel.add_object(image)
        original = image.annotations

        dialog = panel.open_separate_view(edit_annotations=True)
        assert dialog is not None
        locked_items = [
            item
            for item in dialog.get_plot().get_items()
            if isinstance(item, AnnotatedShape) and str(item.title().text()) == "Locked"
        ]
        assert len(locked_items) == 1
        assert locked_items[0].is_readonly()
        dialog.done(QW.QDialog.DialogCode.Rejected)
        assert image.annotations == original

        dialog = panel.open_separate_view(edit_annotations=True)
        assert dialog is not None
        dialog.done(QW.QDialog.DialogCode.Accepted)

        preserved, migrated, preserved_opaque = image.get_annotations()
        assert preserved == canonical
        assert migrated["format"] == "sigima.annotation"
        assert "plotpy_json" not in migrated
        assert preserved_opaque == opaque
        assert image.roi.to_dict() == original_roi


def test_open_separate_view_without_main_plot_item() -> None:
    """Open a separate view when the object has no item in the main plot."""
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        reference = test_data.create_multigaussian_image()
        target = test_data.create_annotated_image()
        panel.add_object(reference)
        panel.add_object(target)

        target_uuid = get_uuid(target)
        panel.plothandler.remove_item(target_uuid)
        assert panel.plothandler.get(target_uuid) is None

        existing_uuids = panel.plothandler.get_existing_oids()
        existing_items = {uuid: panel.plothandler.get(uuid) for uuid in existing_uuids}
        visibility = {uuid: item.isVisible() for uuid, item in existing_items.items()}
        dialog = panel.open_separate_view(oids=[target_uuid])
        assert dialog is not None
        dialog.close()
        QW.QApplication.sendPostedEvents(None, QC.QEvent.Type.DeferredDelete)
        QW.QApplication.processEvents()

        assert panel.plothandler.get_existing_oids() == existing_uuids
        assert all(
            panel.plothandler.get(uuid) is item for uuid, item in existing_items.items()
        )
        assert {
            uuid: panel.plothandler.get(uuid).isVisible() for uuid in existing_uuids
        } == visibility


if __name__ == "__main__":
    test_annotations_unit()
