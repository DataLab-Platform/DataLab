# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Write a workspace saved by a released DataLab, without provenance block.

The file is the legacy-workspace fixture of the provenance tests: run this script
with a released DataLab installed from PyPI in a separate environment, e.g.::

    python -m venv %TEMP%\\dl_release
    %TEMP%\\dl_release\\Scripts\\python -m pip install datalab-platform PyQt5 pytest
    %TEMP%\\dl_release\\Scripts\\python scripts/provenance_legacy_fixture.py OUT.h5

Scenario (signals then images): S0 -> normalize -> S1 -> moving average -> S2;
average of S0 and T -> S3, then T is deleted; statistics on S1; image I with a
rectangular ROI -> Gaussian filter -> J; centroid on I.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main(path: str) -> None:
    """Build the scenario and save it to *path*."""
    # pylint: disable=import-outside-toplevel
    import numpy as np
    import sigima.params as sp
    from sigima.objects import create_image, create_image_roi, create_signal

    import datalab
    from datalab.env import execenv
    from datalab.tests import datalab_test_app_context

    execenv.unattended = True
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        x = np.linspace(0.0, 1.0, 5)
        win.set_current_panel("signal")
        win.add_object(create_signal("S0", x, np.array([-2.0, 0.0, 1.0, 4.0, 3.0])))
        win.select_objects([1])
        win.calc("normalize", sp.NormalizeParam.create(method="maximum"), edit=False)
        win.select_objects([2])
        win.calc("moving_average", sp.MovingAverageParam.create(n=3), edit=False)
        win.add_object(create_signal("T", x, np.array([1.0, 1.0, 2.0, 2.0, 3.0])))
        win.select_objects([1, 4])
        win.calc("average", edit=False)
        win.select_objects([2])
        win.calc("stats", edit=False)
        win.select_objects([4])
        win.signalpanel.remove_object(force=True)
        win.set_current_panel("image")
        image = create_image("I", np.arange(12, dtype=np.float64).reshape(3, 4))
        image.roi = create_image_roi("rectangle", [0, 0, 2, 2])
        win.add_object(image)
        win.select_objects([1])
        win.calc("gaussian_filter", sp.GaussianParam.create(sigma=1.0), edit=False)
        win.select_objects([1])
        win.calc("centroid", edit=False)
        win.save_to_h5_file(path)
    print(f"Written by DataLab {datalab.__version__}: {path}")


if __name__ == "__main__":
    main(sys.argv[1])
