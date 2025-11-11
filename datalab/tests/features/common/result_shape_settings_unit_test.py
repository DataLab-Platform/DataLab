# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit test for visualization settings for annotated shapes and markers.

This test verifies that visualization settings (configured in Settings dialog)
are properly applied when creating geometry results:
1. Centroid produces a marker (MARKER kind from sigima)
2. Contour detection produces annotated shapes (CIRCLE, ELLIPSE, or POLYGON kind)

The test uses the Conf.view.def*.temp() context manager to temporarily modify
settings and verify they're applied to new results.
"""

from __future__ import annotations

from plotpy.items import AnnotatedCircle
from plotpy.styles import MarkerParam, ShapeParam
from sigima.objects import Gauss2DParam, create_image_from_param

from datalab.adapters_metadata import GeometryAdapter
from datalab.adapters_plotpy import GeometryPlotPyAdapter
from datalab.config import Conf
from datalab.env import execenv
from datalab.tests import datalab_test_app_context


def test_ima_shape_param():
    """Test that annotated shape settings are applied to
    enclosing_circle results on images."""
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel

        # Create a Gaussian image for testing (simple peak for enclosing circle)
        SIZE = 200
        param = Gauss2DParam.create(height=SIZE, width=SIZE, sigma=20)
        img = create_image_from_param(param)
        img.title = "Test Gaussian"
        panel.add_object(img)

        # Test: Verify annotated shape settings for enclosing circle
        # ----------------------------------------------------------
        execenv.print("\n=== Test: Enclosing Circle (Annotated Shapes) ===")

        # Create custom annotated shape settings
        def_param = ShapeParam()
        def_param.line.width = 3
        def_param.line.color = "#852727"

        # Temporarily set the annotated shape settings
        # Note: enclosing_circle is an image feature, so we use ima_shape_param
        with Conf.view.ima_shape_param.temp(def_param):
            # Compute enclosing circle with the custom settings
            with Conf.proc.show_result_dialog.temp(False):
                panel.processor.run_feature("enclosing_circle")

            # Get the geometry adapter and create plot items
            adapter = GeometryAdapter.from_obj(img, "enclosing_circle")
            assert adapter is not None, "Enclosing circle should be computed"

            # Create a plotpy adapter to get the shape items
            plotpy_adapter = GeometryPlotPyAdapter(adapter)
            items = list(plotpy_adapter.iterate_shape_items("%.1f", True, "i"))

            # Verify we got annotated shapes
            assert len(items) > 0, "Should have at least one shape"

            shape_item = items[0]
            assert isinstance(shape_item, AnnotatedCircle), (
                f"Expected AnnotatedCircle, got {type(shape_item)}"
            )

            # Verify the annotation settings were applied
            param: ShapeParam = shape_item.shape.shapeparam
            execenv.print(f"Line width: {param.line.width}")
            execenv.print(f"Line color: {param.line.color}")
            assert param.line.width == def_param.line.width, (
                f"Expected line.width={def_param.line.width}, got {param.line.width}"
            )
            assert param.line.color == def_param.line.color, (
                f"Expected line.color={def_param.line.color}, got {param.line.color}"
            )
            execenv.print(
                "✓ Enclosing circle annotated shape settings correctly applied"
            )

        execenv.print("\n=== Test passed ===")


def test_ima_marker_param():
    """Test that shape settings are applied to geometry results."""
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel

        # Create a 2D Gaussian image for testing
        SIZE = 200
        param = Gauss2DParam.create(height=SIZE, width=SIZE, sigma=20)
        img = create_image_from_param(param)
        img.title = "Test Gaussian"
        panel.add_object(img)

        # Test: Verify marker settings for centroid
        # ------------------------------------------
        execenv.print("\n=== Test: Centroid (Marker) ===")

        # Create custom marker settings
        def_param = MarkerParam()
        def_symbol = def_param.symbol
        def_symbol.marker = "XCross"
        def_symbol.size = 15
        def_symbol.edgecolor = "#316331"
        def_symbol.facecolor = "#291b6b"
        def_symbol.alpha = 0.87

        # Temporarily set the marker settings
        with Conf.view.ima_marker_param.temp(def_param):
            # Compute centroid with the custom settings
            with Conf.proc.show_result_dialog.temp(False):
                panel.processor.run_feature("centroid")

            # Get the geometry adapter and create plot items
            adapter = GeometryAdapter.from_obj(img, "centroid")
            assert adapter is not None, "Centroid should be computed"

            # Create a plotpy adapter to get the marker item
            plotpy_adapter = GeometryPlotPyAdapter(adapter)
            items = list(plotpy_adapter.iterate_shape_items("%.1f", True, "i"))

            # Verify we got a marker
            assert len(items) > 0, "Should have at least one marker"
            from plotpy.items import Marker

            marker = items[0]
            assert isinstance(marker, Marker), f"Expected Marker, got {type(marker)}"

            # Verify the marker settings were applied
            symbol = marker.markerparam.symbol
            execenv.print(f"Marker symbol: {symbol.marker}")
            execenv.print(f"Marker size: {symbol.size}")
            execenv.print(f"Marker edge color: {symbol.edgecolor}")
            assert symbol.marker == def_symbol.marker, (
                f"Expected marker='XCross', got '{symbol.marker}'"
            )
            assert symbol.size == def_symbol.size, (
                f"Expected size={def_symbol.size}, got {symbol.size}"
            )
            assert symbol.edgecolor == def_symbol.edgecolor, (
                f"Expected edgecolor={def_symbol.edgecolor}, got {symbol.edgecolor}"
            )
            assert symbol.facecolor == def_symbol.facecolor, (
                f"Expected facecolor={def_symbol.facecolor}, got {symbol.facecolor}"
            )
            execenv.print("✓ Centroid marker settings correctly applied")

        execenv.print("\n=== Test passed ===")


def test_refresh_shape_items_after_settings_change():
    """Test that shape items are refreshed when settings change.

    This test verifies the fix for the issue where result shapes were not
    updated when refreshing the Image View after changing shape
    parameters in the Settings dialog.
    """
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel

        # Create a Gaussian image with a result
        SIZE = 200
        param = Gauss2DParam.create(height=SIZE, width=SIZE, sigma=20)
        img = create_image_from_param(param)
        img.title = "Test Gaussian"
        panel.add_object(img)

        # Compute enclosing circle with initial settings
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("enclosing_circle")

        # Get initial shape item styling
        plot = panel.plothandler.plot
        shape_items = [item for item in plot.items if isinstance(item, AnnotatedCircle)]
        assert len(shape_items) > 0, "Should have at least one shape item"

        initial_width = shape_items[0].shape.shapeparam.line.width
        initial_color = shape_items[0].shape.shapeparam.line.color

        execenv.print(f"Initial shape: width={initial_width}, color={initial_color}")

        # Change temporarily the shape parameters
        new_param = ShapeParam()
        new_param.line.width = 5
        new_param.line.color = "#00ff00"
        with Conf.view.ima_shape_param.temp(new_param):
            # Call refresh_all_shape_items() to apply new settings
            panel.plothandler.refresh_all_shape_items()

            # Get updated shape items
            shape_items_after = [
                item for item in plot.items if isinstance(item, AnnotatedCircle)
            ]
            assert len(shape_items_after) > 0, (
                "Should still have shape items after refresh"
            )

            updated_width = shape_items_after[0].shape.shapeparam.line.width
            updated_color = shape_items_after[0].shape.shapeparam.line.color

            execenv.print(
                f"Updated shape: width={updated_width}, color={updated_color}"
            )

            # Verify the shape was updated with new settings
            assert updated_width == 5, f"Expected width 5, got {updated_width}"
            assert updated_color == "#00ff00", (
                f"Expected color #00ff00, got {updated_color}"
            )

            execenv.print("✓ Shape items correctly refreshed after settings change\n")


if __name__ == "__main__":
    test_ima_shape_param()
    test_ima_marker_param()
    test_refresh_shape_items_after_settings_change()
