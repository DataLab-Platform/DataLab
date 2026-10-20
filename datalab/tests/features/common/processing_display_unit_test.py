# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Display updates after a processing: panel switch and properties panel."""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import sigima.params
import sigima.proc.signal as sips
from sigima.objects import create_image

from datalab.gui.processor.base import extract_processing_parameters
from datalab.objectmodel import get_uuid
from datalab.tests.features.common.provenance_unit_test import add_signal, app, run


def test_history_text_does_not_need_parameters() -> None:
    """The History text is built from the processing chain, not its parameters."""
    with app() as win:
        s0 = add_signal(win)
        (s1,) = run(
            win, sips.addition_constant, sigima.params.ConstantParam.create(value=2.0)
        )
        (s2,) = run(win, sips.absolute)

        light = extract_processing_parameters(s1, parse_param=False)
        full = extract_processing_parameters(s1)
        assert light.param is None and full.param is not None
        assert (light.func_name, light.pattern, light.source_uuid) == (
            full.func_name,
            full.pattern,
            get_uuid(s0),
        )

        objprop = win.signalpanel.objprop
        text = objprop._build_processing_history(s2)  # pylint: disable=W0212
        lines = [line.strip().removeprefix("└─ ") for line in text.splitlines()]
        assert lines[1:] == ["Addition Constant", "Absolute"]


def test_added_object_updates_panel_and_signals_ready_once() -> None:
    """Adding an object shows it at once and signals readiness once.

    Remote clients adding an object wait for ``SIG_READY``: it must be emitted
    exactly once, whether the object is added to the current panel or not.
    """
    with app() as win:
        ready = []
        win.SIG_READY.connect(lambda: ready.append(None))
        s0 = add_signal(win)
        assert win.get_current_panel() == "signal"
        ready.clear()

        image = create_image("I0", np.zeros((4, 4)))
        win.imagepanel.add_object(image)
        assert len(ready) == 1
        assert win.get_current_panel() == "image"
        assert win.imagepanel.objview.get_current_object() is image
        assert win.imagepanel.objprop.properties.dataset.title == "I0"

        win.set_current_panel("signal")
        ready.clear()
        (s1,) = run(win, sips.absolute)
        assert len(ready) == 1
        assert win.signalpanel.objview.get_current_object() is s1
        assert win.signalpanel.objprop.properties.dataset.title == s1.title
        assert s1 is not s0
