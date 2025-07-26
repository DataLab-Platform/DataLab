# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profiling
"""

# guitest: skip

from datalab.env import execenv
from datalab.tests import datalab_test_app_context


def test_profiling():
    """Profiling test"""
    with execenv.context(unattended=True):
        with datalab_test_app_context() as win:
            win.open_h5_files(
                [
                    "C:/Dev/Projets/X-GRID_data/Projets_Oasis/XGRID5/"
                    "VS000001-blobs_doh_profiling.h5"
                ],
                import_all=True,
            )


if __name__ == "__main__":
    test_profiling()
