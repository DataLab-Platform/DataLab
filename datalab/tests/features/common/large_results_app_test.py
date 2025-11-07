# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Large Results Application Test Module

This module contains tests for verifying that DataLab handles large result datasets
efficiently and correctly.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show,skip

import time

import numpy as np
import sigima.enums
import sigima.objects
import sigima.params as sigima_param
from sigima.tests.data import get_test_image

from datalab.adapters_metadata.geometry_adapter import GeometryAdapter
from datalab.config import Conf
from datalab.tests import datalab_test_app_context


def create_polygon_vertices(
    x0: float, y0: float, nb_points: int, radius_mean: float, radius_variation: float
) -> list[tuple[float, float]]:
    """Generate points for a polygon around (x0, y0)

    Args:
        x0: x coordinate of the polygon center
        y0: y coordinate of the polygon center
        nb_points: number of points to generate
        radius_mean: mean radius of the polygon
        radius_variation: variation in radius

    Returns:
        List of (x, y) points representing the polygon vertices
    """
    points = []
    # Calculate number of NaNs to append (random numbers between 0 and nb_points-10):
    num_nans = np.random.randint(0, nb_points - 10)
    for j in range(nb_points - num_nans):
        angle = j * (2 * np.pi / (nb_points - num_nans))
        radius = radius_mean + radius_variation * np.random.rand()
        x = x0 + radius * np.cos(angle)
        y = y0 + radius * np.sin(angle)
        points.append((x, y))
    for _ in range(num_nans):
        points.append((np.nan, np.nan))
    return points


def create_random_polygons(
    size: int, nb_polygons: int, nb_points_per_polygon: int
) -> np.ndarray:
    """Create random polygons

    Args:
        size: size of the area in which to create polygons
        nb_polygons: number of polygons to create
        nb_points_per_polygon: number of points per polygon

    Returns:
        Array of shape (nb_polygons, nb_points_per_polygon, 2)
    """
    polygons = []
    for _ in range(nb_polygons):
        x0 = size * np.random.rand()
        y0 = size * np.random.rand()
        points = create_polygon_vertices(
            x0, y0, nb_points_per_polygon, radius_mean=20, radius_variation=30
        )
        # Append the flattened points:
        polygons.append(points)
    return np.array(polygons)


def test_large_results_scenario(measure_execution_time: bool = False) -> None:
    """Test scenario to verify result truncation limits work correctly

    Args:
        measure_execution_time: if True, measure and print the execution time, then
         quit immediately

    This scenario tests:
    - Contour detection on flower.npy (generates many contours)
    - Shape drawing truncation at max_shapes_to_draw limit
    - Label display truncation at max_cells_in_label limit
    - Performance with large polygons (many points per polygon)

    Performance benchmark (15 polygons × 5000 points):
    - Pure PlotPy: ~473ms (baseline for drawing shapes only)
    - DataLab: ~254ms (includes shape drawing + HTML label generation)
    - Note: DataLab is faster due to optimized HTML truncation before formatting
    """
    nb_polygons = 15
    nb_points_per_polygon = 5000

    with datalab_test_app_context(
        console=False, exec_loop=not measure_execution_time
    ) as win:
        # Create an image panel
        panel = win.imagepanel

        # Load the flower test image
        ima = get_test_image("flower.npy")
        ima.title = "Test image 'flower.npy' - Contour Detection Limit Test"
        panel.add_object(ima)

        # Apply Roberts filter for edge detection
        panel.processor.run_feature("roberts")

        # Run contour detection which should produce a large set of results
        param = sigima_param.ContourShapeParam()
        param.shape = sigima.enums.ContourShape.POLYGON
        with Conf.proc.show_result_dialog.temp(False):
            panel.processor.run_feature("contour_shape", param)

        # Create geometry results manually using many polygons (we generate results
        # in a manner that should be similar to what contour detection would typically
        # produce but in a way that we can control precisely here)
        vertices = create_random_polygons(
            size=ima.data.shape[0],
            nb_polygons=nb_polygons,
            nb_points_per_polygon=nb_points_per_polygon,
        )
        geom_result = sigima.objects.GeometryResult(
            title="Polygon",
            kind=sigima.objects.KindShape.POLYGON,
            coords=vertices.reshape(-1, nb_points_per_polygon * 2),
            func_name="contour_detection_test",
        )
        geom_adapter = GeometryAdapter(geom_result)

        ima2 = ima.copy()
        geom_adapter.add_to(ima2)
        panel.add_object(ima2)

        if measure_execution_time:
            # Now measure the execution time of switching selection between
            # the 3 images:
            # Image #1: flower.npy, as loaded
            # Image #2: Image #1 after Roberts filter + contour detection
            # Image #3: Image #1 with manually added large polygon result
            print("\nMeasuring image switch timings...")
            image_switch_times = {}
            for _j in range(2):  # Doing multiple iterations to stabilize timings
                panel.objview.select_objects([1])
                for i in range(2, 4):
                    start_time = time.perf_counter()
                    panel.objview.select_objects([i])
                    elapsed_time = (time.perf_counter() - start_time) * 1000  # in ms
                    image_switch_times.setdefault(i, []).append(elapsed_time)
                    print(f"  - Switch to image #{i}: {elapsed_time:.1f} ms")
            print("\nImage switch timings (ms):")
            for i in range(2, 4):
                times = image_switch_times[i]
                avg_time = sum(times) / len(times)
                print(
                    f" - Switching to image #{i}: avg {avg_time:.1f} ms "
                    f"over {len(times)} runs"
                )

            # Measure execution time of manual refresh of the image view:
            print("\nMeasuring image view refresh timings...")
            image_refresh_times = {}
            for i in range(3, 4):
                panel.objview.select_objects([i])
                for _j in range(5):  # Doing multiple iterations to stabilize timings
                    start_time = time.perf_counter()
                    panel.manual_refresh()
                    elapsed_time = (time.perf_counter() - start_time) * 1000  # in ms
                    image_refresh_times.setdefault(i, []).append(elapsed_time)
                    print(f"  - Manual refresh: {elapsed_time:.1f} ms")
            print("\nImage view refresh timings (ms):")
            for i in range(3, 4):
                times = image_refresh_times[i]
                avg_time = sum(times) / len(times)
                print(
                    f" - Manual refresh: avg {avg_time:.1f} ms over {len(times)} runs"
                )


if __name__ == "__main__":
    test_large_results_scenario(measure_execution_time=False)
