# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenario D1: full traceability
=======================================

History, provenance and cascade recompute.

The History panel records a processing chain applied to an X-ray diffraction
pattern of paracetamol: noise, smoothing, detrending, peak detection and
multi-Gaussian fit. The smoothing window is then changed upstream, in the
Processing tab of the smoothed signal: the downstream results are recomputed in
cascade, in place. Finally, the provenance of the last execution of the smoothing
shows the Sigima operation, its parameters and its inputs, whose data are checked
against their recorded fingerprints.

The scenario turns the History recording and edit modes on, and shows the History
panel below the signal plot: the previous modes and layout of the main window are
restored at the end.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import sigima.params
import sigima.proc.signal as sips
from guidata.qthelpers import is_qobject_valid
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import NormalDistributionParam
from sigima.tests.data import create_paracetamol_signal
from sigima.tools.signal import fitting
from sigima.tools.signal.peakdetection import peak_indices

from datalab.config import _
from datalab.env import execenv
from datalab.gui.processor.base import extract_processing_parameters
from datalab.gui.showcase.engine import ShowcaseScenario
from datalab.gui.showcase.scenarios.d3_dialogs import drive_dialogs

if TYPE_CHECKING:
    from sigima.objects import SignalObj

    from datalab.gui.panel.history.panel import HistoryPanel
    from datalab.gui.panel.signal import SignalPanel
    from datalab.gui.showcase.engine import ShowcaseContext
    from datalab.history import HistoryAction

#: Size of the moving window of the smoothing: first value, then upstream change
SMOOTHING_WINDOWS = (3, 15)

#: Relative threshold (%) of the peak detection: the default of its dialog
PEAK_THRESHOLD = 50.0

#: Standard deviation of the noise added to the diffraction pattern
NOISE_SIGMA = 20.0

#: Steps recorded in the History panel (function or method names)
RECORDED_STEPS = [
    "new_object",
    "add_gaussian_noise",
    "moving_average",
    "detrending",
    "extract_peak_positions",
    "recompute_fit",
]


@contextmanager
def show_dialogs_longer(
    ctx: ShowcaseContext, seconds: float
) -> Generator[None, None, None]:
    """Show the dialogs longer than usual before accepting them, within the context

    The demo mode of DataLab accepts the dialogs shown with ``exec_dialog`` after a
    short delay: this context lengthens it (the modal-dialog guard of the showcase
    waits for the demo mode to close them). In check mode, nothing is changed.

    Args:
        ctx: Showcase context
        seconds: Delay at normal pace (s)

    Yields:
        None
    """
    previous = execenv.delay
    if not ctx.check:
        execenv.delay = max(previous, int(1000 * seconds * ctx.speed))
    try:
        yield
    finally:
        execenv.delay = previous


def find_panel_action(panel: SignalPanel, text: str) -> QW.QAction:
    """Return an action of a data panel from its text

    Args:
        panel: Data panel
        text: Action text (translated)

    Returns:
        Action

    Raises:
        RuntimeError: If the panel has no such action
    """
    for action in panel.actions():
        if action.text() == text:
            return action
    raise RuntimeError(f"No action {text!r} in the {panel.PANEL_STR}")


def find_tool_button(parent: QW.QWidget, action: QW.QAction) -> QW.QToolButton:
    """Return the toolbar button of an action

    Args:
        parent: Widget containing the toolbar
        action: Action

    Returns:
        Toolbar button

    Raises:
        RuntimeError: If the action has no toolbar button
    """
    for button in parent.findChildren(QW.QToolButton):
        if button.defaultAction() is action:
            return button
    raise RuntimeError(f"No toolbar button for action {action.text()!r}")


def find_item_line_edit(editor: QW.QWidget, name: str) -> QW.QLineEdit:
    """Return the line edit of a data item, in a parameter editor

    Args:
        editor: Parameter editor (``DataSetEditGroupBox``)
        name: Data item name

    Returns:
        Line edit of the data item

    Raises:
        RuntimeError: If the editor has no line edit for this item
    """
    for widget in editor.edit.widgets:
        if widget.item.item.get_name() == name:
            return widget.edit
    raise RuntimeError(f"No editor widget for parameter {name!r}")


def fit_peaks(
    x: np.ndarray, y: np.ndarray, parent: QW.QWidget | None = None
) -> tuple[np.ndarray, list, dict]:
    """Fit a sum of Gaussians to the peaks of a signal, without dialog

    Used in check mode, instead of the interactive multi-Gaussian fit: the peaks
    are those proposed by the peak detection dialog (above half the signal range).

    Args:
        x: X data
        y: Y data
        parent: Parent widget (unused: no dialog)

    Returns:
        Fitted Y data, fit parameters (none) and canonical fit parameters
    """
    del parent  # No dialog
    threshold = np.min(y) + 0.01 * PEAK_THRESHOLD * (np.max(y) - np.min(y))
    indices = peak_indices(y, thres=threshold, min_dist=1, thres_abs=True)
    y_fitted, fit_params = fitting.MultiGaussianFitComputer(x, y, indices).fit()
    return y_fitted, [], fit_params


def select_history_action(
    ctx: ShowcaseContext, hpanel: HistoryPanel, action: HistoryAction
) -> None:
    """Select an action in the History panel tree, by clicking on it

    Args:
        ctx: Showcase context
        hpanel: History panel
        action: Action to select
    """
    tree = hpanel.tree
    iterator = QW.QTreeWidgetItemIterator(tree)
    while iterator.value():
        item = iterator.value()
        if item.data(0, QC.Qt.UserRole) == action.uuid:
            tree.scrollToItem(item)
            rect = tree.visualItemRect(item)
            if rect.isValid():
                viewport = tree.viewport()
                xpos = min(rect.center().x(), viewport.width() // 3)
                ctx.click(viewport, QC.QPoint(xpos, rect.center().y()))
            break
        iterator += 1
    if hpanel.navigation.current_action() is not action:
        hpanel.navigation.select_action_in_tree(action)
    QW.QApplication.processEvents()


def show_history_below_plot(ctx: ShowcaseContext) -> None:
    """Show the History panel below the signal plot, both visible at once

    By default, the History panel is tabified with the signal plot: it is moved
    below it (the layout of the main window is restored at the end of the
    scenario).

    Args:
        ctx: Showcase context
    """
    win = ctx.win
    view_dock = win.docks[win.signalpanel]
    history_dock = win.docks[win.historypanel]
    for dock in (view_dock, history_dock):
        dock.setFloating(False)
        dock.show()
    win.addDockWidget(QC.Qt.RightDockWidgetArea, history_dock, QC.Qt.Vertical)
    win.set_current_panel("signal")
    view_dock.raise_()
    QW.QApplication.processEvents()
    height = view_dock.height() + history_dock.height()
    win.resizeDocks(
        [view_dock, history_dock], [height // 2, height - height // 2], QC.Qt.Vertical
    )
    QW.QApplication.processEvents()


def start_recording(ctx: ShowcaseContext, hpanel: HistoryPanel) -> None:
    """Show the History panel and turn its recording on

    Args:
        ctx: Showcase context
        hpanel: History panel
    """
    show_history_below_plot(ctx)
    dock = ctx.win.docks[hpanel]
    record = hpanel.ui.actions["record"]
    with ctx.spotlight(dock):
        ctx.say(_("The History panel records every action, ready to be replayed"))
        if not hpanel.record_mode_enabled:
            if not record.isChecked():
                ctx.click(find_tool_button(hpanel, record))
            if not hpanel.record_mode_enabled:
                hpanel.toggle_record_mode(True)
            ctx.pause(1.0)
    ctx.checkpoint(hpanel.record_mode_enabled, "History recording is off")


def build_chain(
    ctx: ShowcaseContext, panel: SignalPanel
) -> tuple[SignalObj, SignalObj, SignalObj]:
    """Build the processing chain, with visible menu walks and dialogs

    Args:
        ctx: Showcase context
        panel: Signal panel

    Returns:
        Smoothed, detrended and fitted signals
    """
    # Parameter dialogs are shown (then accepted) at showcase speed only
    edit = None if ctx.check else True
    signal = create_paracetamol_signal(500)
    signal.title = _("Paracetamol")
    ctx.say(_("A raw X-ray diffraction pattern of paracetamol"))
    ctx.win.add_object(signal)
    ctx.pause(1.0)

    ctx.say(_("Simulate a noisy acquisition: add Gaussian noise"), hold=1.5)
    param = NormalDistributionParam.create(mu=0.0, sigma=NOISE_SIGMA, seed=1)
    noisy = ctx.run_feature("add_gaussian_noise", param, panel, edit=edit)
    ctx.checkpoint(noisy is not signal, "No noisy signal")

    window = SMOOTHING_WINDOWS[0]
    ctx.say(_("Smooth it with a moving average (window: %d points)") % window, 1.5)
    param = sigima.params.MovingAverageParam.create(n=window)
    with show_dialogs_longer(ctx, 2.5):
        smoothed = ctx.run_feature("moving_average", param, panel, edit=edit)
    ctx.checkpoint(smoothed is not noisy, "No smoothed signal")

    ctx.say(_("Remove the linear trend of the baseline"), hold=1.5)
    param = sips.DetrendingParam.create(method="linear")
    detrended = ctx.run_feature("detrending", param, panel, edit=edit)
    ctx.checkpoint(detrended is not smoothed, "No detrended signal")

    ctx.say(_("Detect the diffraction peaks"), hold=1.5)
    action = find_panel_action(panel, _("Peak detection..."))
    if ctx.check:
        param = sips.PeakDetectionParam.create(threshold=PEAK_THRESHOLD, min_dist=1)
        panel.processor.compute_peak_detection(param)
    else:
        with show_dialogs_longer(ctx, 2.5):
            ctx.trigger_action(action)
    ctx.pause(1.0)

    ctx.say(_("Fit the peaks with a sum of Gaussians"), hold=1.5)
    title = _("Multi-Gaussian fit")
    if ctx.check:
        panel.processor.compute_fit(title, fit_peaks)
    else:
        with show_dialogs_longer(ctx, 3.5):
            ctx.trigger_action(find_panel_action(panel, title))
    fitted = panel.objview.get_current_object()
    ctx.checkpoint(
        fitted is not detrended and "fit_params" in fitted.metadata,
        "No multi-Gaussian fit",
    )
    panel.objview.select_objects([detrended, fitted])
    ctx.say(_("The fit matches the detected peaks"), hold=2.0)
    return smoothed, detrended, fitted


def show_recorded_chain(
    ctx: ShowcaseContext, hpanel: HistoryPanel
) -> list[HistoryAction]:
    """Walk through the processing chain recorded in the History panel

    Args:
        ctx: Showcase context
        hpanel: History panel

    Returns:
        Recorded actions
    """
    actions = list(hpanel)
    steps = [action.func_name or action.method_name for action in actions]
    ctx.checkpoint(
        steps == RECORDED_STEPS and len(hpanel.history_sessions) == 1,
        f"Unexpected history: {steps}",
    )
    with ctx.spotlight(ctx.win.docks[hpanel]):
        ctx.say(_("The whole chain is recorded: each step, with its parameters"), 2.0)
        hpanel.tree.resizeColumnToContents(0)
        for action in actions:
            select_history_action(ctx, hpanel, action)
            ctx.pause(0.8)
    return actions


def change_upstream_parameter(
    ctx: ShowcaseContext,
    hpanel: HistoryPanel,
    panel: SignalPanel,
    actions: list[HistoryAction],
    smoothed: SignalObj,
    detrended: SignalObj,
) -> None:
    """Change the smoothing window upstream: the downstream results are recomputed

    Args:
        ctx: Showcase context
        hpanel: History panel
        panel: Signal panel
        actions: Recorded actions
        smoothed: Smoothed signal
        detrended: Detrended signal
    """
    window = SMOOTHING_WINDOWS[1]
    ma_action = actions[RECORDED_STEPS.index("moving_average")]
    peaks_action = actions[RECORDED_STEPS.index("extract_peak_positions")]
    peaks_activities = list(peaks_action.provenance_activity_ids)
    detrended_y = detrended.y.copy()
    object_count = len(panel.objmodel)
    # In-place cascade recompute of the downstream actions requires the edit mode
    hpanel.toggle_edit_mode(True)

    ctx.win.set_current_panel("signal")
    objprop = panel.objprop
    panel.objview.select_objects([detrended, smoothed])
    ctx.wait_until(lambda: objprop.current_processing_obj is smoothed, 5.0)
    editor = objprop.processing_param_editor
    ctx.checkpoint(
        objprop.current_processing_obj is smoothed and editor is not None,
        "No Processing tab for the smoothed signal",
    )
    tabs = objprop.tabwidget
    index = tabs.indexOf(objprop.processing_scroll)
    with ctx.spotlight(objprop, ctx.win.signalview):
        ctx.say(
            _("Change a parameter upstream: the smoothing window, from %d to %d points")
            % SMOOTHING_WINDOWS,
            hold=2.0,
        )
        tabbar = tabs.tabBar()
        ctx.click(tabbar, tabbar.tabRect(index).center())
        tabs.setCurrentIndex(index)
        ctx.pause(0.5)
        line_edit = find_item_line_edit(editor, "n")
        line_edit.selectAll()
        ctx.type_text(line_edit, str(window))
        ctx.pause(0.5)
        if editor.apply_button.isEnabled():
            ctx.click(editor.apply_button)
        ctx.wait_until(lambda: not hpanel.runtime.execution.is_busy(), 30.0)
        ctx.pause(0.5)

    with ctx.spotlight(ctx.win.signalview, ctx.win.docks[hpanel]):
        ctx.say(
            _(
                "Detrending and peak detection are recomputed in cascade, "
                "in place: same objects, new data"
            ),
            hold=4.5,
        )

    pparams = extract_processing_parameters(smoothed)
    ctx.checkpoint(
        pparams is not None and pparams.param.n == window,
        "The smoothing window was not changed",
    )
    ctx.checkpoint(
        ma_action.kwargs["param"].n == window,
        "The History action of the smoothing was not updated",
    )
    ctx.checkpoint(
        len(panel.objmodel) == object_count
        and not np.array_equal(detrended.y, detrended_y),
        "The detrended signal was not recomputed in place",
    )
    ctx.checkpoint(
        peaks_action.provenance_activity_ids != peaks_activities,
        "The peak detection was not recomputed",
    )
    ctx.checkpoint(
        not any(action.is_stale for action in actions),
        "Some History actions were not recomputed",
    )


def drive_provenance_dialog(ctx: ShowcaseContext, dialog: QW.QDialog) -> None:
    """Drive the provenance dialog: check its data, then close it

    The "Check data" button compares the data of the inputs and of the results
    with the fingerprints of their recorded states. The dialog is driven by the
    scenario: it is exempted from the modal-dialog guard of the showcase.

    Args:
        ctx: Showcase context
        dialog: Dialog shown by the "Provenance" action of the History panel
    """
    ctx.pause(1.5)
    if dialog.objectName() == "execution_provenance_dialog":
        for button in dialog.findChildren(QW.QPushButton, "provenance_check"):
            if button.isVisible() and button.isEnabled():
                ctx.click(button)
                break
    ctx.pause(3.5)
    if is_qobject_valid(dialog) and dialog.isVisible():
        dialog.accept()


def show_provenance(
    ctx: ShowcaseContext, hpanel: HistoryPanel, action: HistoryAction
) -> None:
    """Show the provenance of the last execution of an action

    Args:
        ctx: Showcase context
        hpanel: History panel
        action: History action
    """
    ledger = ctx.win.provenance.ledger
    calls = []
    for activity_id in action.provenance_activity_ids:
        try:
            calls.append(ledger.activity(activity_id).get("call") or {})
        except KeyError:
            continue
    ctx.checkpoint(
        any(
            (call.get("operation") or {}).get("id") == "sigima.signal.moving_average"
            and (call.get("parameters") or {}).get("n") == SMOOTHING_WINDOWS[1]
            for call in calls
        ),
        "No provenance recorded for the last execution of the smoothing",
    )
    with ctx.spotlight(ctx.win.docks[hpanel]):
        select_history_action(ctx, hpanel, action)
        ctx.say(
            _(
                "Each execution keeps its provenance: Sigima operation, parameters, "
                "inputs and their fingerprints"
            ),
            hold=2.0,
        )
        provenance = hpanel.ui.actions["provenance"]
        if ctx.check:
            ctx.trigger_action(provenance)
        else:
            drive_dialogs(
                ctx,
                lambda: ctx.trigger_action(provenance),
                [partial(drive_provenance_dialog, ctx)],
            )
        ctx.pause(0.5)


def run(ctx: ShowcaseContext) -> None:
    """Play the scenario

    Args:
        ctx: Showcase context
    """
    win = ctx.win
    hpanel = win.historypanel
    panel = win.signalpanel
    record = hpanel.ui.actions["record"]
    state = (
        hpanel.record_mode_enabled,
        record.isChecked(),
        hpanel.is_edit_mode(),
        hpanel.isEnabled(),
    )
    layout = win.saveState()
    try:
        hpanel.setEnabled(True)
        start_recording(ctx, hpanel)
        smoothed, detrended, _fitted = build_chain(ctx, panel)
        actions = show_recorded_chain(ctx, hpanel)
        change_upstream_parameter(ctx, hpanel, panel, actions, smoothed, detrended)
        show_provenance(ctx, hpanel, actions[RECORDED_STEPS.index("moving_average")])
        with ctx.spotlight(win.docks[hpanel], win.signalview):
            ctx.say(
                _("Every processing step is traced, from the raw data to the result"),
                hold=4.0,
            )
    finally:
        record_mode, record_checked, edit_mode, enabled = state
        if hpanel.is_edit_mode() != edit_mode:
            hpanel.toggle_edit_mode(edit_mode)
        record.setChecked(record_checked)
        hpanel.toggle_record_mode(record_mode)
        hpanel.setEnabled(enabled)
        win.restoreState(layout)
        QW.QApplication.processEvents()


SCENARIO = ShowcaseScenario(
    id="d1",
    title=_("Full traceability: history, provenance, cascade recompute"),
    subtitle=_("Every result knows where it comes from"),
    run=run,
    duration_hint=90.0,
)
