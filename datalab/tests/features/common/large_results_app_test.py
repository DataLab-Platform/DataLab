# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

""" """

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show,skip

import time

import numpy as np
import sigima.enums
import sigima.objects
import sigima.params as sigima_param
from sigima.tests.data import get_test_image

from datalab.adapters_metadata.geometry_adapter import GeometryAdapter
from datalab.tests import datalab_test_app_context


def generate_polygon_points(x0, y0, nb_points, radius_mean, radius_variation):
    """Generate points for a polygon around (x0, y0)"""
    points = []
    # Calculate number of NaNs to append (random numbers between 0 and nb_points-10):
    num_nans = np.random.randint(0, nb_points - 10)
    for j in range(nb_points - num_nans):
        angle = j * (2 * np.pi / (nb_points - num_nans))
        radius = radius_mean + radius_variation * np.random.rand()
        x = x0 + radius * np.cos(angle)
        y = y0 + radius * np.sin(angle)
        points.extend((x, y))
    #  Add couple of NaNs
    for _ in range(num_nans):
        points.extend((np.nan, np.nan))
    return points


def generate_random_polygon(size, nb_polygons, nb_points_per_polygon):
    """Generate random polygons"""
    polygons = []
    for _ in range(nb_polygons):
        x0 = size * np.random.rand()
        y0 = size * np.random.rand()
        points = generate_polygon_points(
            x0, y0, nb_points_per_polygon, radius_mean=20, radius_variation=30
        )
        polygons.append(points)
    return polygons


def test_large_results_scenario(measure_execution_time: bool = False) -> None:
    """Test scenario to verify result truncation limits work correctly

    Args:
        measure_execution_time: if True, measure and print the execution time, then

    This scenario tests:
    - Contour detection on flower.npy (generates many contours)
    - Result truncation at max_result_rows limit
    - Shape drawing truncation at max_shapes_to_draw limit
    - Label display truncation at max_cells_in_label limit
    - Warning dialog when displaying large result sets
    """
    with datalab_test_app_context(
        console=False, exec_loop=not measure_execution_time
    ) as win:
        # Create an image panel
        panel = win.imagepanel

        # Load the flower test image
        ima = get_test_image("flower.npy")
        ima.title = "Test image 'flower.npy' - Contour Detection Limit Test"
        ima.set_metadata_option("colormap", "jet")
        panel.add_object(ima)

        # Apply Roberts filter for edge detection
        panel.processor.run_feature("roberts")

        # Run contour detection which should trigger the limits
        # This will detect many contours and test our safety mechanisms
        print("\nRunning contour detection on flower.npy...")
        print("This should trigger result truncation and shape drawing limits.")
        param = sigima_param.ContourShapeParam()
        param.shape = sigima.enums.ContourShape.POLYGON
        panel.processor.run_feature("contour_shape", param)

        # Create geometry results manually using many polygons:
        points = generate_random_polygon(
            size=ima.data.shape[0], nb_polygons=200, nb_points_per_polygon=50
        )
        geom_result = sigima.objects.GeometryResult(
            title="Polygon",
            kind=sigima.objects.KindShape.POLYGON,
            coords=np.array(points),
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


if __name__ == "__main__":
    test_large_results_scenario(measure_execution_time=False)
