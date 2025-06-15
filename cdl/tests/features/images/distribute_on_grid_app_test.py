# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Distribute on grid application test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import sigima_.param
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.data import get_test_image


def test_distribute_on_grid():
    """Run distribute on grid application test scenario"""
    with execenv.context(unattended=True):
        with cdltest_app_context(console=False) as win:
            panel = win.imagepanel
            proc = panel.processor
            panel.add_object(get_test_image("flower.npy"))
            proc.compute_all_morphology(sigima_.param.MorphologyParam.create(radius=10))
            panel.objview.select_groups()

            # Distribute on grid
            params = [
                sigima_.param.GridParam.create(cols=4),
                sigima_.param.GridParam.create(
                    rows=3, colspac=20, rowspac=20, direction="row"
                ),
                sigima_.param.GridParam.create(cols=2, colspac=10, rowspac=10),
            ]
            origins = []
            for param in params:
                proc.distribute_on_grid(param)
                objs = panel.objview.get_sel_objects(include_groups=True)
                origins.append([(obj.x0, obj.y0) for obj in objs])

            # Reset positions
            proc.reset_positions()
            objs = panel.objview.get_sel_objects(include_groups=True)
            origins.append([(obj.x0, obj.y0) for obj in objs])

        assert origins[0][0] == (0.0, 0.0)
        assert origins[0][1] == (512.0, 0.0)
        assert origins[0][2] == (1024.0, 0.0)
        assert origins[0][3] == (1536.0, 0.0)
        assert origins[0][4] == (0.0, 512.0)
        assert origins[0][5] == (512.0, 512.0)
        assert origins[0][6] == (1024.0, 512.0)

        assert origins[1][0] == (0.0, 0.0)
        assert origins[1][1] == (0.0, 532.0)
        assert origins[1][2] == (0.0, 1064.0)
        assert origins[1][3] == (532.0, 0.0)
        assert origins[1][4] == (532.0, 532.0)
        assert origins[1][5] == (532.0, 1064.0)

        assert origins[2][0] == (0.0, 0.0)
        assert origins[2][1] == (522.0, 0.0)
        assert origins[2][2] == (0.0, 522.0)
        assert origins[2][3] == (522.0, 522.0)
        assert origins[2][4] == (0.0, 1044.0)
        assert origins[2][5] == (522.0, 1044.0)

        for x0, y0 in origins[3]:
            assert x0 == 0.0 and y0 == 0.0


if __name__ == "__main__":
    test_distribute_on_grid()
