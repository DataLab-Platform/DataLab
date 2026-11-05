# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""History panel descriptions stay compact for voluminous parameters (issue #369)."""

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
import sigima.params
from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW
from sigima.objects import create_image

from datalab.history.action import HistoryAction
from datalab.history.description import (
    DETAILS_TEXT_MAX_LENGTH,
    DETAILS_VALUE_MAX_LENGTH,
    SUMMARY_MAX_LENGTH,
    SUMMARY_VALUE_MAX_LENGTH,
    shorten,
)
from datalab.widgets.historydescription import (
    CollapsibleDescriptionWidget,
    ElidedLabel,
)


class TextParam(gds.DataSet, title="Text parameters"):
    """Parameters holding a possibly long text beside a short value."""

    text = gds.StringItem("Text", default="")
    count = gds.IntItem("Count", default=3)


def make_action(param: object) -> HistoryAction:
    """Return a compute action recorded with ``param``."""
    return HistoryAction(
        "Operation",
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="image",
        func_name="adjust_brightness_contrast",
        pattern="1_to_1",
        kwargs={"param": param},
    )


def make_brightness_contrast_action() -> HistoryAction:
    """Return a Brightness and contrast action holding its histogram payload."""
    rng = np.random.default_rng(0)
    image = create_image("Source", rng.integers(0, 4000, (64, 64)).astype(np.uint16))
    param = sigima.params.BrightnessContrastParam()
    param.update_from_obj(image)
    param.minimum, param.maximum = 100.0, 3000.0
    assert len(param.histogram["counts"]) == 256
    return make_action(param)


def test_brightness_contrast_description_shows_range() -> None:
    """The histogram payload is replaced by the selected range."""
    action = make_brightness_contrast_action()
    title = action.kwargs["param"].get_title()

    assert action.description_summary == f"{title}: minimum=100, maximum=3000"
    details = action.description_html
    assert "counts" not in details and "bin_edges" not in details
    assert "(minimum)" in details and "(maximum)" in details
    assert "100" in details and "3000" in details
    assert len(details) < 2000


def test_brightness_contrast_description_survives_copy() -> None:
    """Copies (as after saving/reloading) are described like the original."""
    action = make_brightness_contrast_action()
    copy = action.copy()

    assert copy.kwargs["param"].histogram == {}
    assert copy.description_summary == action.description_summary
    assert copy.description_html == action.description_html


def test_long_dataset_value_is_shortened() -> None:
    """Each long value is shortened: the other values remain visible."""
    param = TextParam()
    param.text = "<b>" + "x" * 10000
    action = make_action(param)

    summary = action.description_summary
    assert len(summary) <= SUMMARY_MAX_LENGTH
    expected_text = shorten("<b>" + "x" * 10000, SUMMARY_VALUE_MAX_LENGTH)
    assert summary == f"Text parameters: text={expected_text}, count=3"
    details = action.description_html
    assert "&lt;b&gt;" in details and "<b>x" not in details
    assert "…" in details and "(count)" in details
    assert len(details) < 2 * DETAILS_VALUE_MAX_LENGTH


def test_summary_of_many_values_is_bounded() -> None:
    """The whole summary is bounded, whatever the number of parameters."""
    params = [TextParam() for _index in range(50)]
    for param in params:
        param.text = "y" * 60
    action = make_action(params[0])
    for index, param in enumerate(params[1:]):
        action.kwargs[f"other{index}param"] = param

    summary = action.description_summary
    assert len(summary) <= SUMMARY_MAX_LENGTH
    assert summary.endswith("…")


def test_long_single_line_non_dataset_description_is_bounded() -> None:
    """A long single-line value is shortened in both renderings."""
    action = make_action({"values": list(range(5000))})

    summary = action.description_summary
    assert len(summary) <= SUMMARY_MAX_LENGTH and summary.endswith("…")
    assert "\n" not in summary
    details = action.description_html
    assert details.endswith("…")
    assert len(details) <= DETAILS_TEXT_MAX_LENGTH + 100


def test_short_non_dataset_description_is_unchanged() -> None:
    """Short descriptions keep their first line and their line breaks."""
    action = make_action("first line\nsecond line")

    assert action.description_summary == "first line"
    assert action.description_html == "first line<br>second line"


def test_collapsed_description_is_one_line_high() -> None:
    """A collapsed row is one line high, whatever the summary length."""
    with qt_app_context():
        short = CollapsibleDescriptionWidget("short", "<b>details</b>")
        long_summary = "z" * 10000
        long = CollapsibleDescriptionWidget(long_summary, "<b>details</b>")
        long.resize(400, long.sizeHint().height())
        long.show()
        QW.QApplication.processEvents()

        assert long.sizeHint().height() == short.sizeHint().height()
        assert long.sizeHint().height() < 3 * long.fontMetrics().lineSpacing()
        label = long.findChild(ElidedLabel)
        assert label.full_text() == long_summary
        assert len(label.text()) < len(long_summary)
        assert label.toolTip() == long_summary
        long.close()


def test_expanded_description_height_is_capped() -> None:
    """Long expanded details are scrolled inside a bounded cell."""
    with qt_app_context():
        details = "<br>".join(f"line {index}" for index in range(200))
        widget = CollapsibleDescriptionWidget("summary", details)
        widget.set_expanded(True)
        widget.resize(400, widget.sizeHint().height())
        widget.show()
        QW.QApplication.processEvents()

        assert widget.is_expanded()
        assert widget.sizeHint().height() == widget.details_max_height()
        scroll_area = widget.findChild(QW.QScrollArea)
        assert scroll_area.verticalScrollBar().maximum() > 0
        widget.close()


def test_initially_expanded_description_collapses() -> None:
    """A description restored expanded collapses at the first request."""
    with qt_app_context():
        widget = CollapsibleDescriptionWidget("summary", "<b>details</b>", True)
        assert widget.is_expanded()
        widget.set_expanded(False)
        assert not widget.is_expanded()
