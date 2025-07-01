# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Utilities to manage GUI activation for tests executed with pytest
or as standalone scripts.
"""

from __future__ import annotations

import types

import pytest

CURRENT_REQUEST: pytest.FixtureRequest | None = None


def set_current_request(request: pytest.FixtureRequest | None) -> None:
    """Store the current pytest request object (for use in is_gui_enabled)"""
    global CURRENT_REQUEST  # pylint: disable=global-statement
    CURRENT_REQUEST = request


def is_gui_enabled() -> bool:
    """
    Return True if GUI mode is enabled (i.e. pytest was run with --gui),
    or if a DummyRequest with --gui was set (for __main__ execution).
    """
    return bool(CURRENT_REQUEST and CURRENT_REQUEST.config.getoption("--gui"))


class DummyRequest:
    """
    Dummy request object to simulate pytest --gui when running a test manually.

    Example usage:
        test_x(request=DummyRequest(gui=True))
    """

    def __init__(self, gui: bool = True):
        self.config = types.SimpleNamespace()
        self.config.getoption = lambda name: gui if name == "--gui" else None
