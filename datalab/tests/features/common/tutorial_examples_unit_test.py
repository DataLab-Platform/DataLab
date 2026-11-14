# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Regression tests for the standalone tutorial examples."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage as spi

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    ("parts", "is_kernel"),
    [
        (("doc", "intro", "tutorials", "custom_func.py"), False),
        (("plugins", "examples", "datalab_custom_func.py"), False),
        (("macros", "examples", "imageproc_macro.py"), True),
    ],
)
def test_denoising_examples(parts: tuple[str, ...], is_kernel: bool) -> None:
    """Constant neighborhoods stay constant and nonconstant ones remain finite."""
    filename = ROOT.joinpath(*parts)
    tree = ast.parse(filename.read_text(encoding="utf-8"), filename=str(filename))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "weighted_average_denoise"
    )
    namespace = {"np": np, "spi": spi}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(filename), "exec"),
        namespace,
    )
    denoise = namespace["weighted_average_denoise"]

    def apply(data: np.ndarray) -> np.ndarray:
        return spi.generic_filter(data, denoise, size=5) if is_kernel else denoise(data)

    constant = np.full((7, 7), 3.0)
    impulse = np.zeros((7, 7))
    impulse[3, 3] = 10.0
    with np.errstate(divide="raise", invalid="raise"):
        np.testing.assert_allclose(apply(constant), constant)
        filtered = apply(impulse)
    assert np.isfinite(filtered).all()
    assert filtered[3, 3] == pytest.approx(10 / (1 + 24 * np.exp(-25 / 24)))


@pytest.mark.parametrize("num_points", [100, 101])
def test_spyder_example_has_finite_values(num_points: int) -> None:
    """The first peak has a valid width on both even and odd grids."""
    functions = runpy.run_path(str(ROOT / "doc" / "intro" / "tutorials" / "my_work.py"))
    with np.errstate(divide="raise", invalid="raise"):
        x, data = functions["generate_2d_data"](
            num_lines=2, num_points=num_points, noise_std=0.0
        )
    assert data.shape == (2, num_points)
    assert np.isfinite(data).all()
    np.testing.assert_allclose(data[0], 0.1 * np.sin(10 * x))
    _, expected = functions["generate_1d_data"](
        0.1, 0.5, 0.1, noise=False, num_points=num_points
    )
    np.testing.assert_allclose(data[1], expected)
