# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Tests for DataLab Qt helpers."""

from qtpy import QtWidgets as QW

from datalab.env import execenv
from datalab.utils.qthelpers import block_signals, datalab_app_context


def test_block_signals_restores_nested_states() -> None:
    """Nested blockers restore the state owned by their outer context."""
    with datalab_app_context(exec_loop=False, enable_logs=False):
        parent = QW.QWidget()
        child = QW.QWidget(parent)
        child.blockSignals(True)

        with block_signals(parent, children=True):
            assert parent.signalsBlocked()
            assert child.signalsBlocked()
            with block_signals(parent, children=True):
                assert parent.signalsBlocked()
                assert child.signalsBlocked()
            assert parent.signalsBlocked()
            assert child.signalsBlocked()

        assert not parent.signalsBlocked()
        assert child.signalsBlocked()


def test_no_event_loop_context_leaves_no_close_timer() -> None:
    """A context without ``exec()`` does not close a later top-level widget."""
    with execenv.context(unattended=True):
        with datalab_app_context(exec_loop=False, enable_logs=False) as application:
            pass

        widget = QW.QWidget()
        widget.show()
        application.processEvents()

        assert widget.isVisible()
        widget.close()
