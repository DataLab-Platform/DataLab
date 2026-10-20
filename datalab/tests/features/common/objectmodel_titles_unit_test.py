# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Object numbering and title references in the object model.

Titles refer to other objects by stable ``<UUID8>`` references, so adding,
removing or reordering objects only renumbers them: titles never change.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest
from sigima.objects import create_signal

from datalab.objectmodel import (
    ObjectModel,
    get_short_id,
    get_title_reference,
    get_uuid,
    set_number,
)


def _signal(title: str):
    return create_signal(title, np.arange(3.0), np.arange(3.0))


def _model(count: int) -> tuple[ObjectModel, str, list]:
    model = ObjectModel("gs")
    group = model.add_group("G")
    gid = get_uuid(group)
    objs = []
    for index in range(count):
        # Each object refers to the previous one, as processing titles do
        title = "S0" if index == 0 else f"f({get_title_reference(objs[-1])})"
        obj = _signal(title)
        model.add_object(obj, gid)
        objs.append(obj)
    return model, gid, objs


def test_titles_survive_removal_and_reordering() -> None:
    """Removing or reordering objects renumbers them, titles are unchanged."""
    model, gid, objs = _model(4)
    titles = [o.title for o in objs]
    group = model.get_group(gid)
    group.remove(objs[0])
    model.reset_short_ids()
    assert [get_short_id(o) for o in objs[1:]] == ["s001", "s002", "s003"]
    model.reorder_objects({gid: [get_uuid(o) for o in reversed(objs[1:])]})
    assert [get_short_id(o) for o in objs[1:]] == ["s003", "s002", "s001"]
    assert [o.title for o in objs] == titles


def test_title_reference_index() -> None:
    """The title reference index finds the items ``find_by_title_reference`` does."""
    model, gid, objs = _model(3)
    index = model.get_title_reference_index()
    for item in (*objs, model.get_group(gid)):
        reference = get_title_reference(item)
        assert index[reference] == model.find_by_title_reference(reference)
        assert index[reference] == [item]
    assert "0" * 8 not in index


def test_appending_renumbers_nothing() -> None:
    """Objects added after every other object renumber nothing.

    ``add_objects`` tells whether the short IDs of existing items changed.
    """
    model, gid, objs = _model(3)
    other = model.add_group("H")
    titles = [o.title for o in objs]
    new = _signal(f"f({get_title_reference(objs[2])})")
    # The last group is empty: appending to the first one renumbers nothing
    assert model.add_objects([new], gid) is False
    assert get_short_id(new) == "s004" and [o.title for o in objs] == titles
    # Adding to the last group renumbers nothing either
    last = _signal(f"g({get_title_reference(new)})")
    assert model.add_objects([last], get_uuid(other)) is False
    assert get_short_id(last) == "s005"
    # Inserting before the objects of another group renumbers them
    middle = _signal(f"h({get_title_reference(objs[0])})")
    assert model.add_objects([middle], gid) is True
    assert (get_short_id(middle), get_short_id(last)) == ("s005", "s006")
    assert middle.title == f"h({get_title_reference(objs[0])})"


def test_appending_repairs_inconsistent_numbers() -> None:
    """Numbers out of date are reset as before, and reported as changed."""
    model, gid, objs = _model(3)
    set_number(objs[1], 7)
    new = _signal("new")
    assert model.add_objects([new], gid) is True
    assert [get_short_id(o) for o in [*objs, new]] == ["s001", "s002", "s003", "s004"]


def _replay(general: bool, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Add objects, mostly appended, sometimes inserted before others."""
    if general:
        monkeypatch.setattr(
            ObjectModel, "_ObjectModel__numbering_is_current", lambda _self: False
        )
    rng = np.random.default_rng(3)
    model = ObjectModel("gs")
    groups = [model.add_group("A"), model.add_group("B")]
    for index in range(40):
        obj = _signal(f"op{index}")
        group = groups[0] if rng.random() < 0.2 else groups[1]
        model.add_objects([obj], get_uuid(group))
    monkeypatch.undo()
    return [(get_short_id(o), o.title) for o in model]


def test_appending_matches_general_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The numbers of appended objects are those of the general path."""
    assert _replay(False, monkeypatch) == _replay(True, monkeypatch)
