# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Tests for atomic bulk object operations."""

from __future__ import annotations

import pytest
from sigima.objects import create_signal

from datalab.objectmodel import ObjectModel, get_number, get_title_reference, get_uuid
from datalab.tests import datalab_test_app_context


def create_test_signal(title: str):
    """Create a minimal signal for object-model tests."""
    return create_signal(title, [0.0, 1.0], [0.0, 1.0])


def test_object_model_add_objects_preserves_order_and_references() -> None:
    """Bulk insertion renumbers objects and leaves title references unchanged."""
    model = ObjectModel("gs")
    first_group = model.add_group("First")
    second_group = model.add_group("Second")
    source = create_test_signal("Source")
    model.add_object(source, get_uuid(second_group))
    derived = create_test_signal(f"derived({get_title_reference(source)})")
    model.add_object(derived, get_uuid(second_group))
    derived_title = derived.title
    inserted = (create_test_signal("A"), create_test_signal("B"))

    model.add_objects(inserted, get_uuid(first_group))

    assert model.get_all_objects() == [*inserted, source, derived]
    assert [get_number(obj) for obj in model.get_all_objects()] == [1, 2, 3, 4]
    assert derived.title == derived_title


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
        original_resize = panel.objview.resize_title_column

        def track_update_tree() -> None:
            update_calls.append(None)
            original_update_tree()

        def track_resize(items=None) -> None:
            resize_calls.append(None if items is None else len(items))
            original_resize(items)

        panel.SIG_OBJECT_ADDED.connect(lambda: notifications.append(None))
        monkeypatch.setattr(panel.objview, "update_tree", track_update_tree)
        monkeypatch.setattr(panel.objview, "resize_title_column", track_resize)

        panel.add_objects(signals, get_uuid(group), set_current=False)

        model_ids = panel.objmodel.get_object_ids()
        tree_ids = [
            oid
            for group_ids in panel.objview.get_all_object_uuids().values()
            for oid in group_ids
        ]
        assert model_ids == tree_ids == [get_uuid(signal) for signal in signals]
        assert notifications == [None]
        # The new items are created with their final text: no tree update
        assert not update_calls
        # One column fit, measuring the new items only
        assert resize_calls == [10]
        assert panel.objview.get_current_object() is None

        # Inserting before the objects of another group renumbers them, which
        # changes no item text (titles hold stable references): no tree update
        other = panel.add_group("Other")
        panel.add_objects((create_test_signal("Last"),), get_uuid(other))
        update_calls.clear()
        inserted = tuple(create_test_signal(f"Inserted {i}") for i in range(3))
        panel.add_objects(inserted, get_uuid(group), set_current=False)
        assert not update_calls
        object_ids = panel.objmodel.get_object_ids()
        tree_texts = [panel.objview.get_item_from_id(oid).text(0) for oid in object_ids]
        assert tree_texts == [panel.objmodel[oid].title for oid in object_ids]


def test_adding_an_object_costs_the_same_whatever_the_panel_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding an object measures the new title only, not every title of the panel
    (cost independent of the number of objects)."""
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        panel = win.signalpanel
        first = create_test_signal("Source")
        panel.add_objects((first,))
        reference = get_title_reference(first)
        panel.add_objects(
            tuple(create_test_signal(f"f({reference}) {i}") for i in range(30))
        )
        objview = panel.objview
        hints: list[None] = []
        original_hint = objview.title_delegate.sizeHint

        def track_hint(option, index):
            hints.append(None)
            return original_hint(option, index)

        monkeypatch.setattr(objview.title_delegate, "sizeHint", track_hint)
        counts = []
        for _batch in range(2):
            hints.clear()
            panel.add_objects((create_test_signal(f"g({reference})"),))
            counts.append(len(hints))
            panel.add_objects(
                tuple(create_test_signal(f"f({reference}) {i}") for i in range(60))
            )
        # Same work with 31 and 92 objects: the existing titles are not measured
        assert counts[0] == counts[1] < 31
        before = objview.columnWidth(0)
        panel.add_objects((create_test_signal("A much longer title " * 8),))
        assert objview.columnWidth(0) > before


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
            del set_current
            original_add_items(objects[:1], group_id, set_current=False)
            raise RuntimeError("injected tree failure")

        panel.SIG_OBJECT_ADDED.connect(lambda: notifications.append(None))
        monkeypatch.setattr(panel.objview, "add_object_items", fail_after_first_item)

        with pytest.raises(RuntimeError, match="injected tree failure"):
            panel._add_objects(signals)  # pylint: disable=protected-access

        assert len(panel) == 0
        assert panel.objmodel.get_groups() == []
        assert panel.objview.topLevelItemCount() == 0
        assert not notifications


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
