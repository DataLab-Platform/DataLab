# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Short IDs in titles follow the numbering of the object model.

Titles refer to other objects by short ID (``s001``); the model rewrites them
when objects are added, removed or reordered.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest
from sigima.objects import create_signal

from datalab.objectmodel import ObjectModel, get_short_id, get_uuid, set_number


def _signal(title: str):
    return create_signal(title, np.arange(3.0), np.arange(3.0))


def _model(count: int) -> tuple[ObjectModel, str, list]:
    model = ObjectModel("gs")
    group = model.add_group("G")
    gid = get_uuid(group)
    objs = []
    for index in range(count):
        # Each object refers to the previous one, as processing titles do
        title = "S0" if index == 0 else f"f({get_short_id(objs[-1])})"
        obj = _signal(title)
        model.add_object(obj, gid)
        objs.append(obj)
    return model, gid, objs


def test_titles_follow_removal_and_reordering() -> None:
    """Removing or reordering objects renumbers the references in titles."""
    model, gid, objs = _model(4)
    assert [o.title for o in objs] == ["S0", "f(s001)", "f(s002)", "f(s003)"]
    group = model.get_group(gid)
    group.remove(objs[0])
    assert [get_short_id(o) for o in objs[1:]] == ["s001", "s002", "s003"]
    # The removed object is no longer known: "sxxx"
    assert [o.title for o in objs[1:]] == ["f(sxxx)", "f(s001)", "f(s002)"]
    model.reorder_objects({gid: [get_uuid(o) for o in reversed(objs[1:])]})
    assert [o.title for o in objs[1:]] == ["f(sxxx)", "f(s003)", "f(s002)"]


def test_four_digit_short_ids() -> None:
    """``s100`` is not replaced inside ``s1000`` (more than 999 objects)."""
    model, gid, objs = _model(1001)
    assert objs[1000].title == "f(s1000)"
    assert objs[100].title == "f(s100)"
    model.get_group(gid).remove(objs[0])
    assert objs[1000].title == "f(s999)"
    assert objs[100].title == "f(s099)"


def test_group_and_object_short_ids_are_distinct() -> None:
    """``gs001`` (a group) is not taken for ``s001`` (an object)."""
    model, gid, objs = _model(2)
    obj = _signal("diff(gs001, s002)")
    model.add_object(obj, gid)
    model.add_group("H")
    model.reorder_groups([get_uuid(g) for g in reversed(model.get_groups())])
    assert obj.title == "diff(gs002, s002)"
    assert objs[1].title == "f(s001)"


def test_unknown_uuid_in_title() -> None:
    """A UUID of an object that is not in the model becomes ``sxxx``."""
    model, gid, _objs = _model(1)
    obj = _signal(f"f({get_uuid(_signal('other'))})")
    model.add_object(obj, gid)
    assert obj.title == "f(sxxx)"


def test_short_id_index() -> None:
    """The short ID index finds the same items as ``find_by_short_id``."""
    model, gid, objs = _model(3)
    index = model.get_short_id_index()
    assert index["s002"] is objs[1] is model.find_by_short_id("s002")
    assert index["gs001"] is model.get_group(gid)
    assert "s004" not in index and model.find_by_short_id("s004") is None


def test_appending_leaves_existing_titles() -> None:
    """Objects added after every other object renumber nothing.

    The titles of the new objects are rewritten as when objects are inserted
    before others; ``add_objects`` tells whether existing short IDs changed.
    """
    model, gid, objs = _model(3)
    other = model.add_group("H")
    group = model.get_group(gid)
    unknown = get_uuid(_signal("not in the model"))
    new = _signal(f"f({get_uuid(objs[2])}, {get_uuid(group)}, {unknown}, s001)")
    titles = [o.title for o in objs]
    # The last group is empty: appending to the first one renumbers nothing
    assert model.add_objects([new], gid) is False
    assert new.title == "f(s003, gs001, sxxx, s001)"
    assert get_short_id(new) == "s004" and [o.title for o in objs] == titles
    # Adding to the last group renumbers nothing either
    last = _signal(f"g({get_uuid(new)})")
    assert model.add_objects([last], get_uuid(other)) is False
    assert (get_short_id(last), last.title) == ("s005", "g(s004)")
    # Inserting before the objects of another group renumbers them
    middle = _signal(f"h({get_uuid(objs[0])})")
    assert model.add_objects([middle], gid) is True
    assert (get_short_id(middle), middle.title) == ("s005", "h(s001)")
    assert (get_short_id(last), last.title) == ("s006", "g(s004)")


def test_appending_repairs_inconsistent_numbers() -> None:
    """Numbers out of date are reset as before, and reported as changed."""
    model, gid, objs = _model(3)
    set_number(objs[1], 7)
    new = _signal(f"f({get_uuid(objs[2])})")
    assert model.add_objects([new], gid) is True
    assert [get_short_id(o) for o in [*objs, new]] == ["s001", "s002", "s003", "s004"]
    assert new.title == "f(s003)"


def _replay(general: bool, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Add objects whose titles refer to objects, groups and unknown items."""
    if general:
        monkeypatch.setattr(
            ObjectModel, "_ObjectModel__numbering_is_current", lambda _self: False
        )
    rng = np.random.default_rng(3)
    model = ObjectModel("gs")
    groups = [model.add_group("A"), model.add_group("B")]
    objs = []
    foreign = get_uuid(_signal("foreign"))
    for index in range(40):
        refs = [foreign, get_uuid(groups[int(rng.integers(2))]), "s001", "gs002"]
        if objs:
            refs.append(get_uuid(objs[int(rng.integers(len(objs)))]))
        title = (
            f"op{index}({', '.join(refs[int(k)] for k in rng.permutation(len(refs)))})"
        )
        obj = _signal(title)
        # Mostly appended at the end, sometimes inserted in the first group
        group = groups[0] if rng.random() < 0.2 else groups[1]
        model.add_objects([obj], get_uuid(group))
        objs.append(obj)
    monkeypatch.undo()
    return [(get_short_id(o), o.title) for o in model]


def test_appending_matches_general_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The titles and numbers of appended objects are those of the general path."""
    assert _replay(False, monkeypatch) == _replay(True, monkeypatch)
