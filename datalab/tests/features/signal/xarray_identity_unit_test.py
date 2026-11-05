# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""X-array identity of multi-input signal operations.

Grids of the same size with the same ends are not identical when an inner
coordinate differs: such signals must be interpolated, never combined index by
index.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest
import sigima.proc.signal as sips
from sigima.objects import SignalObj, create_signal

from datalab.config import Conf
from datalab.objectmodel import get_uuid
from datalab.tests.features.common.provenance_unit_test import app

# Same size, same ends, different inner coordinate; y = x + 1 on both grids.
X_SOURCE, Y_SOURCE = [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]
X_OTHER, Y_OTHER = [0.0, 0.5, 2.0], [1.0, 1.5, 3.0]


def add(win, x, y, title: str) -> SignalObj:
    """Add a signal to the signal panel."""
    obj = create_signal(title, np.array(x), np.array(y))
    win.signalpanel.add_object(obj)
    return obj


def new_objects(win, before: set[str]) -> list[SignalObj]:
    """Return the signals added since *before*."""
    model = win.signalpanel.objmodel
    return [model[uid] for uid in model.get_object_ids() if uid not in before]


@pytest.mark.parametrize("pairwise", [False, True])
def test_two_to_one_interpolates_inner_differences(pairwise: bool) -> None:
    """Division: the operand is interpolated, not divided index by index."""
    with app() as win, Conf.xarray_compat_behavior.context("interpolate"):
        source = add(win, X_SOURCE, Y_SOURCE, "source")
        operand = add(win, X_OTHER, Y_OTHER, "operand")
        win.signalpanel.objview.select_objects([get_uuid(source)])
        before = set(win.signalpanel.objmodel.get_object_ids())
        win.signalpanel.processor.compute_2_to_1(
            [operand] if pairwise else operand,
            "operand",
            sips.division,
            edit=False,
            pairwise=pairwise,
        )
        (result,) = new_objects(win, before)
        assert np.array_equal(result.x, X_SOURCE)
        assert np.array_equal(result.y, [1.0, 1.0, 1.0])
        for obj, x, y in ((source, X_SOURCE, Y_SOURCE), (operand, X_OTHER, Y_OTHER)):
            assert np.array_equal(obj.x, x) and np.array_equal(obj.y, y)


def test_n_to_one_interpolates_inner_differences() -> None:
    """Average: the second signal is interpolated, not averaged index by index."""
    with app() as win, Conf.xarray_compat_behavior.context("interpolate"):
        first = add(win, X_SOURCE, Y_SOURCE, "first")
        second = add(win, X_OTHER, Y_OTHER, "second")
        win.signalpanel.objview.select_objects([get_uuid(first), get_uuid(second)])
        before = set(win.signalpanel.objmodel.get_object_ids())
        win.signalpanel.processor.compute_n_to_1(sips.average, edit=False)
        (result,) = new_objects(win, before)
        assert np.array_equal(result.x, X_SOURCE)
        assert np.array_equal(result.y, Y_SOURCE)


def test_identical_grids_are_not_interpolated() -> None:
    """Exactly equal grids keep the signals as they are."""
    with app() as win:
        signals = [
            add(win, X_SOURCE, Y_SOURCE, "a"),
            add(win, X_SOURCE, [2.0, 4.0, 6.0], "b"),
        ]
        checked, yes_to_all = (
            win.signalpanel.processor._check_signal_xarray_compatibility(  # pylint: disable=protected-access
                signals
            )
        )
        assert all(a is b for a, b in zip(checked, signals)) and not yes_to_all
        other = add(win, X_OTHER, Y_OTHER, "c")
        with Conf.xarray_compat_behavior.context("interpolate"):
            checked, _ = win.signalpanel.processor._check_signal_xarray_compatibility(  # pylint: disable=protected-access
                [signals[0], other]
            )
        assert checked[0] is signals[0] and checked[1] is not other
        assert np.array_equal(checked[1].x, X_SOURCE)
