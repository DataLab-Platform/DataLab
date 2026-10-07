# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Tests for atomic bulk object operations."""

from __future__ import annotations

import pytest
from sigima.objects import create_signal

from datalab.objectmodel import ObjectModel, get_number, get_uuid
from datalab.tests import datalab_test_app_context


def create_test_signal(title: str):
    """Create a minimal signal for object-model tests."""
    return create_signal(title, [0.0, 1.0], [0.0, 1.0])


def test_object_model_add_objects_preserves_order_and_references() -> None:
    """Bulk insertion renumbers objects and updates short-ID references once."""
    model = ObjectModel("gs")
    first_group = model.add_group("First")
    second_group = model.add_group("Second")
    source = create_test_signal("Source")
    model.add_object(source, get_uuid(second_group))
    derived = create_test_signal("derived(s001)")
    model.add_object(derived, get_uuid(second_group))
    inserted = (create_test_signal("A"), create_test_signal("B"))

    model.add_objects(inserted, get_uuid(first_group))

    assert model.get_all_objects() == [*inserted, source, derived]
    assert [get_number(obj) for obj in model.get_all_objects()] == [1, 2, 3, 4]
    assert derived.title == "derived(s003)"


def test_object_model_add_objects_rejects_internal_duplicate() -> None:
    """A duplicate inside a batch is rejected before model mutation."""
    model = ObjectModel("gs")
    group = model.add_group("Group")
    signal = create_test_signal("Signal")

    with pytest.raises(ValueError, match="cannot be added twice"):
        model.add_objects((signal, signal), get_uuid(group))

    assert len(model) == 0
    assert group.get_objects() == []
    assert "__number" not in signal.metadata


def test_object_model_add_objects_rejects_existing_uuid() -> None:
    """A UUID collision with the model is rejected before mutation."""
    model = ObjectModel("gs")
    group = model.add_group("Group")
    existing = create_test_signal("Existing")
    model.add_object(existing, get_uuid(group))
    collision = create_test_signal("Collision")
    collision.set_metadata_option("uuid", get_uuid(existing))

    with pytest.raises(ValueError, match="UUID already exists"):
        model.add_objects((collision,), get_uuid(group))

    assert model.get_all_objects() == [existing]
    assert group.get_objects() == [existing]
    assert "__number" not in collision.metadata


def test_object_model_add_objects_rolls_back_partial_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error after insertion restores objects, titles, and numbering."""
    model = ObjectModel("gs")
    group = model.add_group("Group")
    existing = create_test_signal("Existing")
    model.add_object(existing, get_uuid(group))
    group.number = 17
    inserted = create_test_signal("derived(s001)")

    def fail_title_restoration() -> None:
        raise RuntimeError("injected title failure")

    monkeypatch.setattr(
        model,
        "replace_uuids_by_short_ids_in_titles",
        fail_title_restoration,
    )

    with pytest.raises(RuntimeError, match="injected title failure"):
        model.add_objects((inserted,), get_uuid(group))

    assert model.get_all_objects() == [existing]
    assert group.get_objects() == [existing]
    assert existing.title == "Existing"
    assert get_number(existing) == 1
    assert group.number == 17
    assert inserted.title == "derived(s001)"
    assert "__number" not in inserted.metadata


def test_panel_add_objects_updates_tree_and_notifies_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A panel bulk insertion performs one final tree maintenance cycle."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        group = panel.add_group("Campaign")
        signals = tuple(create_test_signal(f"Signal {index}") for index in range(10))
        notifications: list[None] = []
        update_calls: list[None] = []
        resize_calls: list[int] = []
        original_update_tree = panel.objview.update_tree
        original_resize = panel.objview.resizeColumnToContents

        def track_update_tree() -> None:
            update_calls.append(None)
            original_update_tree()

        def track_resize(column: int) -> None:
            resize_calls.append(column)
            original_resize(column)

        panel.SIG_OBJECT_ADDED.connect(lambda: notifications.append(None))
        monkeypatch.setattr(panel.objview, "update_tree", track_update_tree)
        monkeypatch.setattr(panel.objview, "resizeColumnToContents", track_resize)

        panel.add_objects(signals, get_uuid(group), set_current=False)

        model_ids = panel.objmodel.get_object_ids()
        tree_ids = [
            oid
            for group_ids in panel.objview.get_all_object_uuids().values()
            for oid in group_ids
        ]
        assert model_ids == tree_ids == [get_uuid(signal) for signal in signals]
        assert notifications == [None]
        assert update_calls == [None]
        assert resize_calls == [0]
        assert panel.objview.get_current_object() is None


def test_panel_add_objects_sets_last_object_current() -> None:
    """The final object becomes current when bulk insertion requests it."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        signals = tuple(create_test_signal(f"Signal {index}") for index in range(3))

        win.signalpanel.add_objects(signals)

        assert win.signalpanel.objview.get_current_object() is signals[-1]


def test_select_objects_notifies_once_for_final_selection() -> None:
    """Programmatic multi-selection publishes only its complete final state."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        signals = tuple(create_test_signal(f"Signal {index}") for index in range(5))
        win.signalpanel.add_objects(signals, set_current=False)
        notifications: list[tuple] = []
        win.signalpanel.objview.SIG_SELECTION_CHANGED.connect(
            lambda: notifications.append(
                tuple(win.signalpanel.objview.get_sel_objects())
            )
        )

        win.signalpanel.objview.select_objects(signals)

        assert notifications == [signals]
        assert win.signalpanel.objview.get_current_object() is signals[-1]


@pytest.mark.parametrize("selection_kind", ["objects", "numbers", "uuids"])
def test_select_objects_accepts_all_identifier_forms(selection_kind: str) -> None:
    """Object instances, numbers, and UUIDs produce the same final selection."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        signals = tuple(create_test_signal(f"Signal {index}") for index in range(4))
        panel = win.signalpanel
        panel.add_objects(signals, set_current=False)
        selection = {
            "objects": list(signals[1:]),
            "numbers": [2, 3, 4],
            "uuids": [get_uuid(signal) for signal in signals[1:]],
        }[selection_kind]

        panel.objview.select_objects(selection)

        assert panel.objview.get_sel_objects() == list(signals[1:])
        assert panel.objview.get_current_object() is signals[-1]


def test_atomic_selection_preserves_empty_and_group_object_invariants() -> None:
    """Empty calls are no-ops and group/object selections remain exclusive."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        first_group = panel.add_group("First")
        second_group = panel.add_group("Second")
        first = create_test_signal("First signal")
        second = create_test_signal("Second signal")
        panel.add_objects((first,), get_uuid(first_group), set_current=False)
        panel.add_objects((second,), get_uuid(second_group), set_current=False)
        notifications: list[None] = []
        panel.objview.SIG_SELECTION_CHANGED.connect(lambda: notifications.append(None))

        panel.objview.select_objects([first])
        panel.objview.select_objects([])
        assert panel.objview.get_sel_objects() == [first]
        assert notifications == [None]

        panel.objview.select_groups([get_uuid(first_group), get_uuid(second_group)])
        assert panel.objview.get_sel_objects() == []
        assert panel.objview.get_sel_groups() == [first_group, second_group]
        assert panel.objview.get_current_item_id() == get_uuid(second_group)
        assert notifications == [None, None]

        panel.objview.select_groups([])
        assert panel.objview.get_sel_groups() == [first_group, second_group]
        assert notifications == [None, None]

        panel.objview.select_objects([second])
        assert panel.objview.get_sel_groups() == []
        assert panel.objview.get_sel_objects() == [second]
        assert notifications == [None, None, None]


def test_panel_add_objects_rolls_back_tree_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial tree failure removes the model batch and implicit group."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        signals = tuple(create_test_signal(f"Signal {index}") for index in range(3))
        notifications: list[None] = []
        original_add_items = panel.objview.add_object_items

        def fail_after_first_item(objects, group_id, set_current=True) -> None:
            original_add_items(objects[:1], group_id, set_current=False)
            raise RuntimeError("injected tree failure")

        panel.SIG_OBJECT_ADDED.connect(lambda: notifications.append(None))
        monkeypatch.setattr(panel.objview, "add_object_items", fail_after_first_item)

        with pytest.raises(RuntimeError, match="injected tree failure"):
            panel._add_objects(signals)  # pylint: disable=protected-access

        assert len(panel) == 0
        assert panel.objmodel.get_groups() == []
        assert panel.objview.topLevelItemCount() == 0
        assert notifications == []


def test_refresh_plot_batches_one_hundred_curves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large selection refreshes PlotPy observers and canvas only once."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        signals = tuple(create_test_signal(f"Signal {index}") for index in range(100))
        panel.add_objects(signals, set_current=False)
        plot = panel.plothandler.plot
        item_notifications: list[None] = []
        active_notifications: list[None] = []
        replot_calls: list[None] = []
        original_replot = plot.replot

        def track_replot() -> None:
            replot_calls.append(None)
            original_replot()

        plot.SIG_ITEMS_CHANGED.connect(lambda _plot: item_notifications.append(None))
        plot.SIG_ACTIVE_ITEM_CHANGED.connect(
            lambda _plot: active_notifications.append(None)
        )
        monkeypatch.setattr(plot, "replot", track_replot)

        panel.objview.select_objects(signals)

        items = [panel.plothandler.get(get_uuid(signal)) for signal in signals]
        assert all(item is not None and item.isVisible() for item in items)
        assert plot.get_active_item() is items[-1]
        assert item_notifications == [None]
        assert active_notifications == [None]
        assert replot_calls == [None]
