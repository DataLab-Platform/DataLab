# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Distribute on grid application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
import sigima.params
from sigima.tests.data import get_test_image

from datalab.env import execenv
from datalab.tests import datalab_test_app_context


def get_image_origin(obj):
    """Get image origin (x0, y0) handling both uniform and non-uniform coordinates.

    Args:
        obj: Image object

    Returns:
        Tuple (x0, y0)
    """
    if obj.is_uniform_coords:
        return obj.x0, obj.y0
    else:
        return obj.xcoords[0], obj.ycoords[0]


def test_distribute_on_grid():
    """Run distribute on grid application test scenario"""
    with execenv.context(unattended=True):
        with datalab_test_app_context(console=False) as win:
            panel = win.imagepanel
            proc = panel.processor
            panel.add_object(get_test_image("flower.npy"))
            proc.compute_all_morphology(sigima.params.MorphologyParam.create(radius=10))
            panel.objview.select_groups()

            # Distribute on grid
            params = [
                sigima.params.GridParam.create(cols=4),
                sigima.params.GridParam.create(
                    rows=3, colspac=20, rowspac=20, direction="row"
                ),
                sigima.params.GridParam.create(cols=2, colspac=10, rowspac=10),
            ]
            origins = []
            for param in params:
                proc.distribute_on_grid(param)
                objs = panel.objview.get_sel_objects(include_groups=True)
                origins.append([get_image_origin(obj) for obj in objs])

            # Reset positions
            proc.reset_positions()
            objs = panel.objview.get_sel_objects(include_groups=True)
            origins.append([get_image_origin(obj) for obj in objs])

        # Verify results with tolerance for floating point comparisons
        tol = 1e-10

        # First distribution (cols=4)
        assert np.allclose(origins[0][0], (0.0, 0.0), atol=tol)
        assert np.allclose(origins[0][1], (512.0, 0.0), atol=tol)
        assert np.allclose(origins[0][2], (1024.0, 0.0), atol=tol)
        assert np.allclose(origins[0][3], (1536.0, 0.0), atol=tol)
        assert np.allclose(origins[0][4], (0.0, 512.0), atol=tol)
        assert np.allclose(origins[0][5], (512.0, 512.0), atol=tol)
        assert np.allclose(origins[0][6], (1024.0, 512.0), atol=tol)

        # Second distribution (rows=3, colspac=20, rowspac=20, direction="row")
        assert np.allclose(origins[1][0], (0.0, 0.0), atol=tol)
        assert np.allclose(origins[1][1], (0.0, 532.0), atol=tol)
        assert np.allclose(origins[1][2], (0.0, 1064.0), atol=tol)
        assert np.allclose(origins[1][3], (532.0, 0.0), atol=tol)
        assert np.allclose(origins[1][4], (532.0, 532.0), atol=tol)
        assert np.allclose(origins[1][5], (532.0, 1064.0), atol=tol)

        # Third distribution (cols=2, colspac=10, rowspac=10)
        assert np.allclose(origins[2][0], (0.0, 0.0), atol=tol)
        assert np.allclose(origins[2][1], (522.0, 0.0), atol=tol)
        assert np.allclose(origins[2][2], (0.0, 522.0), atol=tol)
        assert np.allclose(origins[2][3], (522.0, 522.0), atol=tol)
        assert np.allclose(origins[2][4], (0.0, 1044.0), atol=tol)
        assert np.allclose(origins[2][5], (522.0, 1044.0), atol=tol)

        # After reset, all should be at origin
        for x0, y0 in origins[3]:
            assert np.allclose([x0, y0], [0.0, 0.0], atol=tol)


if __name__ == "__main__":
    test_distribute_on_grid()
