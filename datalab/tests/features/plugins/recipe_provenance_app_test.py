# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Provenance of the results of a plugin recipe (Desktop).

The workspace provenance does not record recipe runs: the Provenance tab of a
recipe result shows the record of the run held by the object (its recipe,
parameters, inputs and other results), and a processing applied to the result
leads back to this run.
"""

from __future__ import annotations

import os.path as osp

import guidata.dataset as gds
import sigima.params
import sigima.proc.signal as sips
from qtpy import QtWidgets as QW

from datalab.gui.plugins.recipe_runner import RecipeRunner
from datalab.objectmodel import get_uuid
from datalab.plugins.recipes import (
    RecipeCardinality,
    RecipeDescriptor,
    RecipeInputSlot,
    RecipeObjectOutput,
    RecipeObjectType,
    RecipeOutcome,
)
from datalab.tests import helpers
from datalab.tests.features.common.provenance_inspection_unit_test import (
    link,
    view_of,
)
from datalab.tests.features.common.provenance_processing_unit_test import (
    add_signal,
    compute,
)

# The autouse fixture is imported to apply to this module too.
# pylint: disable-next=unused-import
from datalab.tests.features.common.provenance_unit_test import (  # noqa: F401
    Y,
    app,
    isolated_param_defaults,
)

RECIPE_ID = "org.example.recipes:dark-current"


class DarkParameters(gds.DataSet):
    """Parameters of the test recipe."""

    gain = gds.FloatItem("Gain", default=1.0)


def _run_recipe(inputs, parameters, _context) -> RecipeOutcome:
    frames = inputs["frames"]
    dark_map = frames[0].copy()
    dark_map.title = "Dark map"
    dark_map.y = (frames[0].y + frames[1].y) * parameters.gain
    hot_pixels = frames[1].copy()
    hot_pixels.title = "Hot pixels"
    return RecipeOutcome(
        objects=(
            RecipeObjectOutput("dark-map", dark_map),
            RecipeObjectOutput("hot-pixels", hot_pixels),
        )
    )


def run_recipe(win) -> tuple:
    """Run the test recipe on two frames; return the frames and the outputs."""
    frames = (add_signal(win, Y, "Frame 1"), add_signal(win, Y * 2.0, "Frame 2"))
    descriptor = RecipeDescriptor(
        recipe_id=RECIPE_ID,
        plugin_version="1.2.0",
        title="Dark current",
        version="1.0.0",
        inputs=(
            RecipeInputSlot("frames", RecipeObjectType.SIGNAL, RecipeCardinality.MANY),
        ),
        parameter_class=DarkParameters,
        run=_run_recipe,
    )
    parameters = DarkParameters()
    parameters.gain = 2.0
    outcome = RecipeRunner(win).run(descriptor, {"frames": frames}, parameters)
    return frames, tuple(output.value for output in outcome.objects)


def test_recipe_result_shows_its_recipe_run() -> None:
    """A recipe result shows the run that produced it in the Provenance tab."""
    with app() as win:
        frames, (dark_map, hot_pixels) = run_recipe(win)
        view = view_of(win, dark_map)
        objprop = win.signalpanel.objprop
        index = objprop.tabwidget.indexOf(objprop.history_stack)
        assert objprop.tabwidget.isTabVisible(index)
        assert view.has_content()
        assert (
            "the application method that produced it recorded its run"
            in view.summary.text()
        )
        text = view.text()
        assert "Method run" in text
        assert f"Method: {RECIPE_ID}, version 1.0.0" in text
        assert "gain = 2.0" in text
        assert "Inputs (2)" in text
        assert "1. frames: " in text and "Frame 1" in text
        assert "2. frames: " in text and "Frame 2" in text
        assert "dark-map: " in text and "(this object)" in text
        assert "hot-pixels: " in text and "Hot pixels" in text
        assert not view.verify_button.isEnabled()
        gotos = [entry["payload"] for entry in view.links if entry["action"] == "goto"]
        assert gotos == [get_uuid(frames[0]), get_uuid(frames[1]), get_uuid(hot_pixels)]
        view.activate_link(link(view, "details", "d/rr"))
        assert "Run ID: " in view.text()
        goto = link(view, "goto", get_uuid(frames[1]))
        view.activate_link(goto)
        QW.QApplication.processEvents()
        assert win.signalpanel.objview.get_current_object() is frames[1]
        # Consultation does not change the metadata of the inputs
        assert all("__recipe_run_record" not in frame.metadata for frame in frames)


def test_copy_of_a_recipe_result_has_no_recipe_run() -> None:
    """A duplicated recipe result is not attributed to the recipe."""
    with app() as win:
        _frames, (dark_map, _hot_pixels) = run_recipe(win)
        panel = win.signalpanel
        panel.objview.select_objects([get_uuid(dark_map)])
        before = set(panel.objmodel.get_object_ids())
        panel.duplicate_object()
        (copy_uuid,) = set(panel.objmodel.get_object_ids()) - before
        copy = panel.objmodel[copy_uuid]
        panel.objview.set_current_object(copy)
        QW.QApplication.processEvents()
        assert "Method run" not in panel.objprop.provenance_widget.text()


def test_recipe_run_is_shown_after_workspace_round_trip() -> None:
    """The recipe run is still shown after saving and opening the workspace."""
    with helpers.WorkdirRestoringTempDir() as tmpdir, app() as win:
        _frames, (dark_map, _hot_pixels) = run_recipe(win)
        uuid = get_uuid(dark_map)
        filename = osp.join(tmpdir, "recipe-provenance.h5")
        win.save_to_h5_file(filename)
        win.open_h5_files([filename], import_all=True, reset_all=True)
        view = view_of(win, win.find_object_by_uuid(uuid))
        assert f"Method: {RECIPE_ID}, version 1.0.0" in view.text()
        assert "Frame 1" in view.text()


def test_processed_recipe_result_leads_back_to_the_recipe_run() -> None:
    """The input of a processing, made by a recipe, shows its recipe run."""
    with app() as win:
        _frames, (dark_map, _hot_pixels) = run_recipe(win)
        (filtered,) = compute(
            win.signalpanel,
            [dark_map],
            "compute_1_to_1",
            sips.gaussian_filter,
            param=sigima.params.GaussianParam.create(sigma=2.0),
            edit=False,
        )
        view = view_of(win, filtered)
        assert view.description["production"] is not None
        assert "Method run" not in view.text()
        view.activate_link(link(view, "expand", "p/0/r"))
        text = view.text()
        assert f"Method: {RECIPE_ID}, version 1.0.0" in text
        assert "Frame 2" in text
        view.activate_link(link(view, "collapse", "p/0/r"))
        assert RECIPE_ID not in view.text()
