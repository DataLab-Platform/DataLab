# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenario D3: image analysis
====================================

Advanced, interactive image analysis, as a user would do it:

1. 2D peak detection on a noisy image, with automatic regions of interest
2. Contour detection (ellipses) within each region of interest, results table and
   results plot
3. Live preview of a low-pass filter, while its cut-off frequency is dragged, and
   the Processing tab of the result
4. Radial profile of a ring, and full width at half maximum of the ring
5. Line profile extraction, picked on the image

At showcase speed, the parameter dialogs are driven as a user would do (see
:mod:`datalab.gui.showcase.scenarios.d3_dialogs`). In check mode, the features are
run with explicit parameters, without any dialog.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import sigima.objects
import sigima.params
from qtpy import QtWidgets as QW
from sigima.enums import ContourShape
from sigima.tests.data import (
    PeakDataParam,
    RingParam,
    create_ring_image,
    get_peak2d_data,
)

from datalab.adapters_metadata import GeometryAdapter, TableAdapter
from datalab.adapters_metadata.common import create_resultdata_dict
from datalab.config import Conf, _
from datalab.gui.processor.base import extract_processing_parameters
from datalab.gui.showcase.engine import ShowcaseScenario, plain_text
from datalab.gui.showcase.scenarios.d3_dialogs import (
    choose,
    click_button,
    click_plot,
    dialog_delay,
    drag_slider,
    drive_dialogs,
    enable_preview,
    find_item_widget,
    get_edit_layouts,
    set_check_box,
    type_value,
    warm_up_preview,
    zoom_plot,
)

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.showcase.engine import ShowcaseContext
    from datalab.widgets.processingpreview import ProcessingPreviewDialog

#: Seeds of the random data, for a reproducible showcase
PEAKS_SEED, RING_SEED = 7, 1

#: Cut-off frequency ratio of the low-pass filter, chosen with the live preview
CUT_OFF = 0.05

#: Style of the signals plotting the coordinates of the peaks (signal metadata):
#: points, without line
POINTS_STYLE = {"style": "NoPen", "marker": "Ellipse", "size": 10}

#: Parameter classes of the features run by the scenario: their dialogs start
#: from the default values (the values remembered by DataLab are restored after)
PARAM_CLASSES = (
    sigima.params.Peak2DDetectionParam,
    sigima.params.ContourShapeParam,
    sigima.params.ButterworthParam,
    sigima.params.RadialProfileParam,
    sigima.params.FWHMParam,
    sigima.params.LineProfileParam,
)


def create_peaks_image() -> ImageObj:
    """Create a noisy image with bright 2D peaks

    As :func:`sigima.tests.data.create_peak_image`, with larger peaks (so that
    their contours are visible) and reproducible random data.

    Returns:
        Image object
    """
    param = PeakDataParam.create(
        size=512, num_peaks=5, sigma_gauss2d=0.2, sigma_noise=60
    )
    data, coords = get_peak2d_data(param, seed=PEAKS_SEED)
    obj = sigima.objects.create_image(_("Noisy 2D peaks"), data)
    obj.metadata["peak_coords"] = coords
    return obj


def create_noisy_ring_image() -> ImageObj:
    """Create a noisy ring image, from :func:`sigima.tests.data.create_ring_image`

    Returns:
        Image object
    """
    param = RingParam.create(
        image_size=400, xc=200, yc=200, radius=120, thickness=8, intensity=1000
    )
    obj = create_ring_image(param)
    rng = np.random.default_rng(RING_SEED)
    obj.data = obj.data.astype(np.float64) + rng.normal(0.0, 150.0, obj.data.shape)
    obj.title = _("Noisy ring")
    return obj


def find_action(panel: BaseDataPanel, text: str) -> QW.QAction:
    """Return the action of a data panel from its text

    Args:
        panel: Data panel
        text: Text of the action (translated with ``_()``)

    Returns:
        Action

    Raises:
        LookupError: If the panel has no such action
    """
    text = plain_text(text)
    for action in panel.actions():
        if plain_text(action.text()) == text:
            return action
    raise LookupError(f"No action {text!r} in the {panel.PANEL_STR_ID} panel")


def get_result_titles(obj: SignalObj | ImageObj) -> list[str]:
    """Return the titles of the analysis results of an object

    Args:
        obj: Signal or image object

    Returns:
        Titles of the results (e.g. ``'peak'``, ``'contour'``)
    """
    return [
        adapter.title
        for adapter_class in (GeometryAdapter, TableAdapter)
        for adapter in adapter_class.iterate_from_obj(obj)
    ]


@contextmanager
def showcase_settings(ctx: ShowcaseContext) -> Generator[None, None, None]:
    """Set the global state used by the scenario, then restore it

    The results dialog is not shown automatically after each analysis, and the
    parameter dialogs start from the default values: the current panel, the
    parameters remembered by DataLab and the settings are restored afterwards, and
    the windows left open are closed.

    Args:
        ctx: Showcase context

    Yields:
        None
    """
    win = ctx.win
    panel = ctx.current_panel()
    defaults = win.imagepanel.processor.PARAM_DEFAULTS
    remembered = dict(defaults)
    for paramclass in PARAM_CLASSES:
        defaults.pop(paramclass.__name__, None)
    try:
        with Conf.show_result_dialog.context(False):
            yield
    finally:
        defaults.clear()
        defaults.update(remembered)
        ctx.close_stray_windows()
        win.set_current_panel(panel)
        QW.QApplication.processEvents()


def detect_peaks(ctx: ShowcaseContext) -> ImageObj:
    """Step 1: 2D peak detection, with a region of interest around each peak

    Args:
        ctx: Showcase context

    Returns:
        Image of the peaks
    """
    win, panel = ctx.win, ctx.win.imagepanel
    win.set_current_panel("image")
    image = create_peaks_image()
    win.add_object(image)
    ctx.pause(1.0)
    ctx.checkpoint(len(panel.objmodel) == 1, "The image of the peaks was not added")
    ctx.say(_("A noisy image with bright spots: let us find them automatically"))
    action = find_action(panel, _("2D peak detection"))
    if ctx.check:
        param = sigima.params.Peak2DDetectionParam.create(create_rois=True, size=50)
        panel.processor.compute_peak_detection(param)
    else:
        ctx.say(
            _("2D peak detection, with a region of interest around each peak"),
            hold=0.5,
        )

        def create_rois(dialog: QW.QDialog) -> None:
            set_check_box(ctx, dialog, "create_rois")
            click_button(ctx, dialog)

        drive_dialogs(ctx, lambda: ctx.trigger_action(action), [create_rois])
    ctx.checkpoint("peak" in get_result_titles(image), "No 2D peak detection result")
    ctx.checkpoint(
        image.roi is not None and len(image.roi) >= 2,
        "No region of interest was created around the peaks",
    )
    with ctx.spotlight(panel.plothandler.plotwidget):
        ctx.say(
            _("Each peak is detected, and a region of interest is created around it"),
            hold=3.5,
        )
    return image


def fit_contours(ctx: ShowcaseContext, image: ImageObj) -> None:
    """Step 2: contour detection within each region of interest, then results

    Args:
        ctx: Showcase context
        image: Image of the peaks, with regions of interest
    """
    win, panel = ctx.win, ctx.win.imagepanel
    ctx.say(_("Within each region of interest, the spot contour is fitted"), hold=0.5)
    action = ctx.find_feature_action("contour_shape", panel)
    ctx.checkpoint(action is not None, "Contour detection has no action")
    if ctx.check:
        param = sigima.params.ContourShapeParam.create(shape=ContourShape.ELLIPSE)
        ctx.run_feature("contour_shape", param, panel)
    else:

        def fit_ellipses(dialog: QW.QDialog) -> None:
            choose(ctx, dialog, "shape", ContourShape.ELLIPSE)
            click_button(ctx, dialog)

        drive_dialogs(ctx, lambda: ctx.trigger_action(action), [fit_ellipses])
    ctx.checkpoint("contour" in get_result_titles(image), "No contour detection result")
    with ctx.spotlight(panel.plothandler.plotwidget):
        ctx.say(
            _("An ellipse is fitted on each spot, and shown on the image"), hold=3.5
        )

    # Results table
    rdatadict = create_resultdata_dict([image])
    ctx.checkpoint(
        len(rdatadict) >= 2
        and all(len(rdata.results) > 0 for rdata in rdatadict.values()),
        "The results tables are missing",
    )
    ctx.say(_("The results are gathered in tables, ready to be exported"), hold=0.5)
    with dialog_delay(3.0 * ctx.speed):
        ctx.trigger_action(find_action(panel, _("Show results") + "..."))

    # Results plot
    count = len(win.signalpanel.objmodel)
    action = find_action(panel, _("Plot results") + "...")
    if ctx.check:
        panel.plot_results("one_curve_per_object", "x", "y")
    else:
        ctx.say(_("Results may also be plotted: e.g. the peak coordinates"), hold=0.5)

        def plot_coordinates(dialog: QW.QDialog) -> None:
            widget = find_item_widget(get_edit_layouts(dialog), "xaxis")
            keys = [
                choice[0] for choice in widget.item.get_prop_value("data", "choices")
            ]
            if "a" in keys:
                # Ellipse parameters: one point per region of interest, not plotted
                click_button(ctx, dialog, QW.QDialogButtonBox.Cancel)
            else:
                choose(ctx, dialog, "xaxis", "x")
                choose(ctx, dialog, "yaxis", "y")
                click_button(ctx, dialog)

        drive_dialogs(
            ctx,
            lambda: ctx.trigger_action(action),
            [plot_coordinates, plot_coordinates],
        )
    ctx.checkpoint(
        len(win.signalpanel.objmodel) > count, "Plotting the results created no signal"
    )
    # The coordinates of the peaks are not ordered: they are shown as points
    for signal in win.signalpanel.objmodel.get_all_objects()[count:]:
        signal.metadata.update(POINTS_STYLE)
    if ctx.current_panel() is not win.signalpanel:
        win.set_current_panel("signal")
    win.signalpanel.refresh_plot("selected", True, True)
    with ctx.spotlight(win.signalpanel.plothandler.plotwidget):
        ctx.say(_("Results become signals, for further processing"), hold=3.0)


def preview_filter(ctx: ShowcaseContext) -> ImageObj:
    """Step 3: live preview of a low-pass filter, then its Processing tab

    Args:
        ctx: Showcase context

    Returns:
        Filtered ring image
    """
    win, panel = ctx.win, ctx.win.imagepanel
    win.set_current_panel("image")
    ring = create_noisy_ring_image()
    win.add_object(ring)
    ctx.pause(1.0)
    ctx.say(_("A noisy ring: a low-pass filter will smooth it out"))
    action = ctx.find_feature_action("butterworth", panel)
    ctx.checkpoint(action is not None, "Butterworth filter has no action")
    updates = 0
    if ctx.check:
        param = sigima.params.ButterworthParam.create(cut_off=CUT_OFF)
        ctx.run_feature("butterworth", param, panel)
    else:
        ctx.say(
            _("Its live preview follows the cut-off frequency, while it is dragged"),
            hold=0.5,
        )

        def tune_cut_off(dialog: ProcessingPreviewDialog) -> None:
            nonlocal updates
            watcher = enable_preview(ctx, dialog)
            ctx.pause(1.0)
            # The slider moves the cut-off frequency by steps of 0.1...
            count = watcher.count
            widget = find_item_widget(get_edit_layouts(dialog), "cut_off")
            drag_slider(ctx, widget, [0.5, 0.1])
            watcher.wait_for_update(ctx, since=count)
            # ...then the value is fine-tuned by typing it
            count = watcher.count
            type_value(ctx, dialog, "cut_off", str(CUT_OFF))
            watcher.wait_for_update(ctx, since=count)
            ctx.pause(1.5)
            updates = watcher.count
            click_button(ctx, dialog)

        drive_dialogs(ctx, lambda: ctx.trigger_action(action), [tune_cut_off])
        ctx.checkpoint(updates >= 3, "The live preview was not updated while dragging")
    result = panel.objview.get_current_object()
    pparam = extract_processing_parameters(result)
    ctx.checkpoint(
        result is not ring
        and pparam is not None
        and pparam.param is not None
        and abs(pparam.param.cut_off - CUT_OFF) < 1e-9,
        "The filter was not applied with the cut-off frequency of the preview",
    )
    objprop = panel.objprop
    ctx.checkpoint(
        objprop.processing_scroll is not None
        and objprop.processing_param_editor is not None,
        "The Processing tab of the filtered image is missing",
    )
    objprop.tabwidget.setCurrentWidget(objprop.processing_scroll)
    with ctx.spotlight(objprop):
        widget = find_item_widget(objprop.processing_param_editor.edit, "cut_off")
        ctx.move_cursor_to(widget.edit)
        ctx.say(
            _(
                "Parameters stay attached to the result: "
                "they may be changed later, in the Processing tab"
            ),
            hold=4.0,
        )
    return result


def measure_ring(ctx: ShowcaseContext, ring: ImageObj) -> None:
    """Step 4: radial profile of the ring, then its full width at half maximum

    Args:
        ctx: Showcase context
        ring: Filtered ring image
    """
    win, panel = ctx.win, ctx.win.imagepanel
    sigpanel = win.signalpanel
    ctx.say(_("The radial profile of the ring is computed around its centroid"), 0.5)
    count = len(sigpanel.objmodel)
    action = find_action(panel, _("Radial profile extraction..."))
    if ctx.check:
        param = sigima.params.RadialProfileParam()
        param.update_from_obj(ring)
        panel.processor.compute_radial_profile(param)
    else:

        def preview_profile(dialog: ProcessingPreviewDialog) -> None:
            enable_preview(ctx, dialog)
            ctx.pause(1.5)
            click_button(ctx, dialog)

        drive_dialogs(ctx, lambda: ctx.trigger_action(action), [preview_profile])
    ctx.checkpoint(len(sigpanel.objmodel) == count + 1, "No radial profile signal")
    profile = sigpanel.objview.get_current_object()
    if ctx.current_panel() is not sigpanel:
        win.set_current_panel("signal")
    ctx.pause(1.0)

    ctx.say(_("Its full width at half maximum is measured, on one side"), 0.5)
    action = ctx.find_feature_action("fwhm", sigpanel)
    ctx.checkpoint(action is not None, "FWHM has no action")
    if ctx.check:
        ctx.run_feature("fwhm", sigima.params.FWHMParam.create(xmin=0.0), sigpanel)
    else:

        def measure_width(dialog: QW.QDialog) -> None:
            type_value(ctx, dialog, "xmin", "0")
            click_button(ctx, dialog)

        drive_dialogs(ctx, lambda: ctx.trigger_action(action), [measure_width])
    ctx.checkpoint("fwhm" in get_result_titles(profile), "No FWHM result")
    # Zoom on the outer side of the ring, where the width is measured
    outer = profile.x > 0
    xpeak = float(profile.x[outer][np.argmax(profile.y[outer])])
    with ctx.spotlight(sigpanel.plothandler.plotwidget):
        if not ctx.check:
            zoom_plot(ctx, sigpanel.plothandler.plot, xpeak - 40.0, xpeak + 40.0)
        ctx.say(
            _("The width of the ring is measured, and annotated on the curve"),
            hold=4.0,
        )


def extract_line_profile(ctx: ShowcaseContext, ring: ImageObj) -> None:
    """Step 5: line profile picked on the image, extracted to the Signal Panel

    Args:
        ctx: Showcase context
        ring: Filtered ring image
    """
    win, panel = ctx.win, ctx.win.imagepanel
    sigpanel = win.signalpanel
    win.set_current_panel("image")
    panel.objview.set_current_object(ring)
    ctx.pause(1.0)
    ctx.say(_("Finally, a line profile is picked right on the image"), hold=0.5)
    count = len(sigpanel.objmodel)
    action = find_action(panel, _("Line profile..."))
    if ctx.check:
        param = sigima.params.LineProfileParam.create(
            direction="horizontal", x=ring.xc, y=ring.yc
        )
        panel.processor.compute_line_profile(param)
    else:

        def pick_line(dialog: QW.QDialog) -> None:
            ctx.pause(0.8)
            click_plot(ctx, dialog.get_plot(), ring.xc, ring.yc)
            ctx.pause(2.5)
            click_button(ctx, dialog)

        drive_dialogs(ctx, lambda: ctx.trigger_action(action), [pick_line])
    ctx.checkpoint(len(sigpanel.objmodel) == count + 1, "No line profile signal")
    profile = sigpanel.objview.get_current_object()
    ctx.checkpoint(profile.x.size > 0, "The line profile is empty")
    if ctx.current_panel() is not sigpanel:
        win.set_current_panel("signal")
    with ctx.spotlight(sigpanel.plothandler.plotwidget, sigpanel.objview):
        ctx.say(_("The profile lands in the Signal Panel, ready for analysis"), 3.5)


def run(ctx: ShowcaseContext) -> None:
    """Play the scenario

    Args:
        ctx: Showcase context
    """
    with showcase_settings(ctx):
        if not ctx.check:
            warm_up_preview(ctx.win)
        image = detect_peaks(ctx)
        fit_contours(ctx, image)
        ring = preview_filter(ctx)
        measure_ring(ctx, ring)
        extract_line_profile(ctx, ring)
        ctx.say(_("Detect, measure, preview, extract: all interactive, all traced"))


SCENARIO = ShowcaseScenario(
    id="d3",
    title=_("Advanced, interactive image analysis"),
    subtitle=_("Measure, detect and annotate images, right on the plot"),
    run=run,
    duration_hint=105.0,
)
