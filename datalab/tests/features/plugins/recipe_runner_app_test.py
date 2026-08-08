# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Application tests for transactional plugin recipe execution."""

from __future__ import annotations

import json
import os.path as osp

import guidata.dataset as gds
import numpy as np
import pytest
from sigima.objects import (
    GeometryResult,
    KindShape,
    TableResult,
    create_image,
    create_signal,
)

from datalab.adapters_metadata import GeometryAdapter, TableAdapter
from datalab.gui.recipe_runner import RecipeCommitError, RecipeRunner
from datalab.objectmodel import get_uuid
from datalab.recipes import (
    RECIPE_RUN_RECORD_OPTION,
    RecipeCancellationError,
    RecipeCardinality,
    RecipeDescriptor,
    RecipeExecutionContext,
    RecipeInputSlot,
    RecipeObjectOutput,
    RecipeObjectType,
    RecipeOutcome,
    RecipeResultOutput,
    RecipeRunRecord,
    RecipeRunStatus,
    RecipeValidationError,
)
from datalab.tests import datalab_test_app_context, helpers


class RunnerParameters(gds.DataSet):
    """Minimal parameters for runner validation tests."""

    gain = gds.FloatItem("Gain", default=1.0)


def test_recipe_runner_commits_cross_panel_outputs_and_anchored_results() -> None:
    """A successful recipe is committed only after headless execution completes."""
    source = create_signal("Source", x=[0.0, 1.0], y=[1.0, 2.0])

    with datalab_test_app_context(console=False) as win:

        def run_recipe(inputs, _parameters, _context) -> RecipeOutcome:
            assert len(win.signalpanel) == 0
            assert len(win.imagepanel) == 0
            signal = inputs["source"][0].copy()
            signal.title = "Checked signal"
            image = create_image("Diagnostic image", data=np.ones((2, 2)))
            table = TableResult(
                title="Summary metrics",
                headers=["Mean"],
                data=[[1.5]],
            )
            limits = TableResult(
                title="Acceptance limits",
                headers=["Minimum", "Maximum"],
                data=[[1.0, 2.0]],
            )
            geometry = GeometryResult(
                title="Detected point",
                kind=KindShape.POINT,
                coords=np.array([[0.5, 0.5]]),
            )
            return RecipeOutcome(
                objects=(
                    RecipeObjectOutput("summary", signal),
                    RecipeObjectOutput("diagnostic", image),
                ),
                results=(
                    RecipeResultOutput("metrics", table, anchor_id="summary"),
                    RecipeResultOutput("limits", limits, anchor_id="summary"),
                    RecipeResultOutput(
                        "detected-point",
                        geometry,
                        anchor_id="diagnostic",
                    ),
                ),
            )

        descriptor = RecipeDescriptor(
            recipe_id="org.example.recipes:quick-check",
            plugin_version="2.0.0",
            title="Quick check",
            version="1.0.0",
            inputs=(
                RecipeInputSlot(
                    "source",
                    RecipeObjectType.SIGNAL,
                    RecipeCardinality.ONE,
                ),
            ),
            parameter_class=RunnerParameters,
            run=run_recipe,
        )

        parameters = RunnerParameters()
        parameters.gain = 2.0
        outcome = RecipeRunner(win).run(
            descriptor,
            {"source": (source,)},
            parameters,
        )

        assert len(win.signalpanel) == 1
        assert len(win.imagepanel) == 1
        assert [group.title for group in win.signalpanel.objmodel.get_groups()] == [
            "Quick check"
        ]
        assert [group.title for group in win.imagepanel.objmodel.get_groups()] == [
            "Quick check"
        ]
        adapters = list(TableAdapter.iterate_from_obj(outcome.objects[0].value))
        assert {adapter.func_name for adapter in adapters} == {
            "org.example.recipes:quick-check:limits",
            "org.example.recipes:quick-check:metrics",
        }
        assert all(adapter.get_param().gain == 2.0 for adapter in adapters)
        geometry_adapter = next(
            GeometryAdapter.iterate_from_obj(outcome.objects[1].value)
        )
        assert geometry_adapter.func_name == (
            "org.example.recipes:quick-check:detected-point"
        )
        assert geometry_adapter.get_param().gain == 2.0
        output_objects = [output.value for output in outcome.objects]
        records = [
            RecipeRunRecord.from_dict(obj.get_metadata_option(RECIPE_RUN_RECORD_OPTION))
            for obj in output_objects
        ]
        assert records[0] == records[1]
        assert records[0].plugin_version == "2.0.0"
        assert records[0].recipe_version == "1.0.0"
        assert records[0].status is RecipeRunStatus.COMPLETED
        assert records[0].input_uuids == {"source": (get_uuid(source),)}
        assert records[0].output_uuids == {
            output.id: get_uuid(output.value) for output in outcome.objects
        }
        assert gds.json_to_dataset(records[0].parameters_json).gain == 2.0
        assert win.is_modified()


def test_recipe_runner_validates_inputs_before_execution() -> None:
    """Invalid slot assignments neither call recipe code nor mutate the workspace."""
    called = False

    def run_recipe(*_args) -> RecipeOutcome:
        nonlocal called
        called = True
        return RecipeOutcome()

    descriptor = RecipeDescriptor(
        recipe_id="org.example.recipes:requires-image",
        plugin_version="1.0.0",
        title="Requires image",
        version="1.0.0",
        inputs=(RecipeInputSlot("source", "image", "one"),),
        parameter_class=RunnerParameters,
        run=run_recipe,
    )

    with datalab_test_app_context(console=False) as win:
        with pytest.raises(RecipeValidationError, match="source"):
            RecipeRunner(win).run(descriptor, {}, RunnerParameters())
        with pytest.raises(RecipeValidationError, match="RunnerParameters"):
            RecipeRunner(win).run(
                descriptor,
                {"source": (create_image("Input", np.ones((2, 2))),)},
            )
        with pytest.raises(RecipeValidationError, match="group title"):
            RecipeRunner(win).run(
                descriptor,
                {"source": (create_image("Input", np.ones((2, 2))),)},
                RunnerParameters(),
                group_title="",
            )

        assert not called
        assert len(win.signalpanel) == 0
        assert len(win.imagepanel) == 0
        assert not win.is_modified()


def test_recipe_runner_rolls_back_partial_cross_panel_commit(monkeypatch) -> None:
    """A failure in the second panel restores models, trees, groups, and dirty state."""

    def run_recipe(*_args) -> RecipeOutcome:
        return RecipeOutcome(
            objects=(
                RecipeObjectOutput("signal", create_signal("Signal", [0.0], [1.0])),
                RecipeObjectOutput("image", create_image("Image", np.ones((2, 2)))),
            )
        )

    descriptor = RecipeDescriptor(
        recipe_id="org.example.recipes:rollback",
        plugin_version="1.0.0",
        title="Rollback",
        version="1.0.0",
        run=run_recipe,
    )

    with datalab_test_app_context(console=False) as win:
        existing_group = win.signalpanel.add_group("Existing")
        existing_signal = create_signal("Existing", [0.0], [0.0])
        win.signalpanel.add_object(existing_signal, existing_group.uuid)
        win.set_modified(False)

        def fail_image_add(*_args, **_kwargs) -> None:
            raise RuntimeError("injected image commit failure")

        monkeypatch.setattr(win.imagepanel, "_add_object", fail_image_add)

        with pytest.raises(RecipeCommitError, match="injected image commit failure"):
            RecipeRunner(win).run(descriptor, {})

        assert len(win.signalpanel) == 1
        assert len(win.imagepanel) == 0
        assert win.signalpanel.objmodel.get_groups() == [existing_group]
        assert win.imagepanel.objmodel.get_groups() == []
        assert win.signalpanel.objview.topLevelItemCount() == 1
        assert win.imagepanel.objview.topLevelItemCount() == 0
        assert not win.is_modified()


def test_recipe_runner_checks_cancellation_immediately_before_commit() -> None:
    """Cancellation requested during preparation prevents workspace mutation."""
    checks = 0

    def is_cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    def run_recipe(*_args) -> RecipeOutcome:
        return RecipeOutcome(
            objects=(
                RecipeObjectOutput("signal", create_signal("Signal", [0.0], [1.0])),
            )
        )

    descriptor = RecipeDescriptor(
        recipe_id="org.example.recipes:cancel",
        plugin_version="1.0.0",
        title="Cancel",
        version="1.0.0",
        run=run_recipe,
    )
    context = RecipeExecutionContext(cancellation_callback=is_cancelled)

    with datalab_test_app_context(console=False) as win:
        with pytest.raises(RecipeCancellationError):
            RecipeRunner(win).run(descriptor, {}, context=context)

        assert len(win.signalpanel) == 0
        assert win.signalpanel.objmodel.get_groups() == []
        assert not win.is_modified()


def test_recipe_runner_finishes_rollback_after_plot_cleanup_error(monkeypatch) -> None:
    """Model and tree compensation continues when plot cleanup raises."""

    def run_recipe(*_args) -> RecipeOutcome:
        return RecipeOutcome(
            objects=(
                RecipeObjectOutput("signal", create_signal("Signal", [0.0], [1.0])),
                RecipeObjectOutput("image", create_image("Image", np.ones((2, 2)))),
            )
        )

    descriptor = RecipeDescriptor(
        recipe_id="org.example.recipes:cleanup",
        plugin_version="1.0.0",
        title="Cleanup",
        version="1.0.0",
        run=run_recipe,
    )

    with datalab_test_app_context(console=False) as win:

        def fail_image_add(*_args, **_kwargs) -> None:
            raise RuntimeError("injected image commit failure")

        def fail_plot_cleanup(*_args, **_kwargs) -> None:
            raise RuntimeError("injected plot cleanup failure")

        monkeypatch.setattr(win.imagepanel, "_add_object", fail_image_add)
        monkeypatch.setattr(
            win.signalpanel.plothandler,
            "remove_item",
            fail_plot_cleanup,
        )

        with pytest.raises(RecipeCommitError, match="rollback also failed"):
            RecipeRunner(win).run(descriptor, {})

        assert len(win.signalpanel) == 0
        assert len(win.imagepanel) == 0
        assert win.signalpanel.objmodel.get_groups() == []
        assert win.imagepanel.objmodel.get_groups() == []
        assert win.signalpanel.objview.topLevelItemCount() == 0
        assert win.imagepanel.objview.topLevelItemCount() == 0
        assert not win.is_modified()


def test_recipe_runner_rejects_cross_panel_uuid_collision() -> None:
    """Output UUIDs remain globally unique across signal and image panels."""
    existing_image = create_image("Existing", np.ones((2, 2)))

    def run_recipe(*_args) -> RecipeOutcome:
        signal = create_signal("Colliding signal", [0.0], [1.0])
        signal.set_metadata_option("uuid", get_uuid(existing_image))
        return RecipeOutcome(objects=(RecipeObjectOutput("signal", signal),))

    descriptor = RecipeDescriptor(
        recipe_id="org.example.recipes:uuid-collision",
        plugin_version="1.0.0",
        title="UUID collision",
        version="1.0.0",
        run=run_recipe,
    )

    with datalab_test_app_context(console=False) as win:
        win.imagepanel.add_object(existing_image)
        win.set_modified(False)

        with pytest.raises(RecipeValidationError, match="already belongs"):
            RecipeRunner(win).run(descriptor, {})

        assert len(win.signalpanel) == 0
        assert len(win.imagepanel) == 1
        assert win.imagepanel.objmodel[get_uuid(existing_image)] is existing_image
        assert not win.is_modified()


def test_transactional_panel_add_rejects_group_before_model_mutation() -> None:
    """An invalid target group never leaves an orphaned object in the model."""
    signal = create_signal("Signal", [0.0], [1.0])

    with datalab_test_app_context(console=False) as win:
        with pytest.raises(KeyError, match="missing-group"):
            win.signalpanel._add_object(  # pylint: disable=protected-access
                signal, group_id="missing-group"
            )

        assert len(win.signalpanel) == 0
        assert not win.signalpanel.objmodel.has_uuid(get_uuid(signal))
        assert win.signalpanel.objview.topLevelItemCount() == 0
        assert not win.is_modified()


def test_recipe_run_record_survives_workspace_round_trip() -> None:
    """Recipe provenance and referenced UUIDs survive HDF5 persistence."""

    def run_recipe(inputs, _parameters, _context) -> RecipeOutcome:
        output = inputs["source"][0].copy()
        output.title = "Output"
        return RecipeOutcome(objects=(RecipeObjectOutput("output", output),))

    descriptor = RecipeDescriptor(
        recipe_id="org.example.recipes:persistence",
        plugin_version="3.2.1",
        title="Persistence",
        version="1.1.0",
        inputs=(RecipeInputSlot("source", "signal", "one"),),
        parameter_class=RunnerParameters,
        run=run_recipe,
    )

    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            source = create_signal("Source", [0.0, 1.0], [1.0, 2.0])
            win.signalpanel.add_object(source)
            parameters = RunnerParameters()
            parameters.gain = 4.0
            outcome = RecipeRunner(win).run(
                descriptor,
                {"source": (source,)},
                parameters,
            )
            output = outcome.objects[0].value
            output_uuid = get_uuid(output)
            payload = output.get_metadata_option(RECIPE_RUN_RECORD_OPTION)

            filename = osp.join(tmpdir, "recipe-provenance.h5")
            win.save_to_h5_file(filename)
            win.open_h5_files([filename], import_all=True, reset_all=True)

            loaded_output = win.find_object_by_uuid(output_uuid)
            assert loaded_output is not None
            assert (
                loaded_output.get_metadata_option(RECIPE_RUN_RECORD_OPTION) == payload
            )
            loaded_record = RecipeRunRecord.from_dict(payload)
            assert loaded_record.input_uuids == {"source": (get_uuid(source),)}
            assert loaded_record.output_uuids == {"output": output_uuid}


def test_recipe_run_record_serializes_parameterless_recipe() -> None:
    """A parameterless recipe records the resolved JSON null value."""

    def run_recipe(_inputs, parameters, _context) -> RecipeOutcome:
        assert parameters is None
        output = create_signal("Output", [0.0], [1.0])
        return RecipeOutcome(objects=(RecipeObjectOutput("output", output),))

    descriptor = RecipeDescriptor(
        recipe_id="org.example.recipes:parameterless",
        plugin_version="1.0.0",
        title="Parameterless",
        version="1.0.0",
        run=run_recipe,
    )

    with datalab_test_app_context(console=False) as win:
        outcome = RecipeRunner(win).run(descriptor, {})
        payload = outcome.objects[0].value.get_metadata_option(RECIPE_RUN_RECORD_OPTION)
        record = RecipeRunRecord.from_dict(payload)
        assert record.parameters_json == "null"
        assert json.loads(record.parameters_json) is None
