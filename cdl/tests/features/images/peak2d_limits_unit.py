# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image peak detection test: testing algorithm limits
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: skip

from guidata.qthelpers import qt_app_context

from cdl.env import execenv
from cdl.tests.features.images.peak2d_unit_test import exec_image_peak_detection_func
from sigima_.algorithms.image import get_2d_peaks_coords
from sigima_.tests.data import get_peak2d_data


def peak2d_limit_test():
    """2D peak detection test"""
    with qt_app_context():
        execenv.print("Testing peak detection algorithm with random generated data:")
        for idx in range(100):
            execenv.print(f"  Iteration #{idx:02d}: ", end="")
            generated_data = get_peak2d_data(multi=True)
            coords = get_2d_peaks_coords(generated_data)
            if coords.shape[0] != 4:
                execenv.print(f"KO - {coords.shape[0]}/4 peaks were detected")
                exec_image_peak_detection_func(generated_data)
            else:
                execenv.print("OK")
        # Showing results for last generated sample
        exec_image_peak_detection_func(generated_data)


if __name__ == "__main__":
    peak2d_limit_test()
