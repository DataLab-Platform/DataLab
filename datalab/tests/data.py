# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Reproducible data used by DataLab integration tests."""

from __future__ import annotations

from sigima.objects import ImageObj, create_image
from sigima.tests.data import PeakDataParam, get_peak2d_data


def create_reproducible_peak_image(seed: int = 1) -> ImageObj:
    """Create a reproducible 1024-pixel image with four peaks.

    Args:
        seed: Seed for the peak positions and background noise.

    Returns:
        Image with the generated peak coordinates in its metadata.
    """
    param = PeakDataParam.create(size=1024)
    data, coords = get_peak2d_data(param, seed=seed)
    image = create_image("Test image (2D peaks)", data)
    image.metadata["peak_coords"] = coords
    return image
