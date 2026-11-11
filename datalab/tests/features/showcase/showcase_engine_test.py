# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase engine tests

Waiting loops, captions, modal-dialog guard and warm-up of the processing worker.
"""

from __future__ import annotations

import pytest
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.env import execenv
from datalab.gui.processor.base import WorkerState
from datalab.gui.showcase.engine import ShowcaseContext, process_events
from datalab.tests import datalab_test_app_context


@pytest.mark.parametrize("speed", [0.0, 0.01])
def test_waits_delete_the_objects_scheduled_for_deletion(speed: float) -> None:
    """Like the main event loop, the waits delete the objects scheduled for deletion

    The showcase never returns to the main event loop: replaced widgets would stay
    painted over the new ones.
    """
    with datalab_test_app_context(console=False) as win:
        ctx = ShowcaseContext(win, speed)
        deleted: list[str] = []
        for name, wait in (
            ("pause", lambda: ctx.pause(0.5)),
            ("wait_until", lambda: ctx.wait_until(lambda: "wait_until" in deleted, 1)),
        ):
            widget = QW.QWidget(win)
            widget.destroyed.connect(lambda _obj=None, name=name: deleted.append(name))
            widget.deleteLater()
            wait()
            assert name in deleted


def test_caption_moves_above_a_window_covering_it() -> None:
    """The caption moves to the top of the main window while a window covers it"""
    with datalab_test_app_context(console=False) as win:
        with ShowcaseContext(win, 0.01) as ctx:
            ctx.say("Caption", hold=0.0)
            caption = ctx.caption_banner
            assert caption is not None and caption.isVisible()
            assert not caption.moved
            bottom = caption.geometry()
            dialog = QW.QDialog(win)
            dialog.setGeometry(
                QC.QRect(win.mapToGlobal(bottom.topLeft()), bottom.size())
            )
            dialog.show()
            try:
                ctx.pause(60.0)  # 0.6 s: the guard places the caption
                assert caption.placement == (caption.top_limit(), None, None)
                assert caption.geometry().bottom() < bottom.top()
                ctx.say("Next caption", hold=0.0)
                assert caption.moved
            finally:
                dialog.close()
            ctx.pause(60.0)
            assert not caption.moved
            assert caption.geometry() == bottom


def test_caption_is_narrowed_beside_a_tall_window() -> None:
    """Beside a window covering the main window from top to bottom, the caption is
    narrowed, on the side with the most room"""
    with datalab_test_app_context(console=False) as win:
        win.resize(1400, 900)
        with ShowcaseContext(win, 0.01) as ctx:
            ctx.say("A caption long enough to be wrapped once narrowed", hold=0.0)
            caption = ctx.caption_banner
            dialog = QW.QDialog(win)
            dialog.setGeometry(
                QC.QRect(win.mapToGlobal(QC.QPoint(500, 0)), QC.QSize(300, 900))
            )
            dialog.show()
            try:
                ctx.pause(60.0)
                geometry = caption.geometry()
                assert caption.moved and geometry.width() < caption.default_width()
                frame = dialog.frameGeometry()
                frame.moveTopLeft(win.mapFromGlobal(frame.topLeft()))
                assert not frame.intersects(geometry)
                assert geometry.left() > frame.right()  # More room on the right
            finally:
                dialog.close()
            ctx.pause(60.0)
            assert not caption.moved


def test_tall_dialog_is_moved_below_the_caption() -> None:
    """A tall modal dialog covering the caption wherever it goes is moved down,
    when the main window has room for both"""
    with datalab_test_app_context(console=False) as win:
        win.resize(1200, 1000)
        with ShowcaseContext(win, 0.01) as ctx:
            ctx.say("Caption", hold=0.0)
            caption = ctx.caption_banner
            bottom = caption.geometry_at()
            top = caption.geometry_at(caption.top_limit())
            # Covering half of the caption at the top, all of it at the bottom
            dialog_top = top.top() + top.height() // 2
            dialog = QW.QDialog(win)
            dialog.setModal(True)
            dialog.setGeometry(
                QC.QRect(
                    win.mapToGlobal(QC.QPoint(bottom.left(), dialog_top)),
                    QC.QSize(bottom.width(), bottom.bottom() - dialog_top),
                )
            )
            dialog.show()
            try:
                ctx.pause(60.0)
                assert caption.placement == (caption.top_limit(), None, None)
                frame = dialog.frameGeometry()
                frame.moveTopLeft(win.mapFromGlobal(frame.topLeft()))
                assert not frame.intersects(caption.geometry())
                assert frame.bottom() < win.height()
            finally:
                dialog.close()


def test_guard_waits_for_the_demo_mode() -> None:
    """The guard never accepts a dialog before the demo mode closes it, but accepts
    a message box after a delay following the showcase speed"""
    with datalab_test_app_context(console=False) as win:
        dialog = QW.QDialog(win)
        box = QW.QMessageBox(win)
        delay = execenv.delay
        try:
            for speed in (0.7, 1.0, 1.5):
                ctx = ShowcaseContext(win, speed)
                assert ctx.modal_timeout == pytest.approx(max(4.0, 6.0 * speed))
                execenv.delay = int(5000 * speed)
                grace = ctx.GUARD_GRACE * max(speed, 1.0)
                assert ctx.guard_timeout(dialog) == pytest.approx(
                    max(ctx.modal_timeout, 5.0 * speed + grace)
                )
                assert ctx.guard_timeout(box) == pytest.approx(ctx.modal_timeout)
                execenv.delay = 0
                assert ctx.guard_timeout(dialog) == pytest.approx(ctx.modal_timeout)
        finally:
            execenv.delay = delay
            dialog.deleteLater()
            box.deleteLater()
            process_events()


def test_processing_worker_warm_up() -> None:
    """At showcase speed, the processing worker warms up while the first title card
    is shown, so that the first computation of a scenario does not wait for it"""
    with datalab_test_app_context(console=False) as win:
        worker = win.signalpanel.processor.worker
        if worker is None:
            pytest.skip("Process isolation is disabled")
        ShowcaseContext(win, 0.0).start_warm_up()
        assert worker.state_machine.current_state == WorkerState.IDLE
        with ShowcaseContext(win, 0.01) as ctx:
            ctx.start_warm_up()
            assert worker.state_machine.current_state == WorkerState.RUNNING
            ctx.title_card("Title", hold=0.0)
            assert worker.state_machine.current_state == WorkerState.IDLE
        # The worker is ready for the computations of the scenarios
        win.signalpanel.new_object()
        win.signalpanel.processor.run_feature("absolute")
        assert len(win.signalpanel) == 2
