# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image FFT unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import numpy as np
import pytest

import sigima_.algorithms.image as alg
import sigima_.computation.image as sigima_image
import sigima_.obj
import sigima_.param
import sigima_.tests.data as ctd
from sigima_.env import execenv
from sigima_.tests.helpers import check_array_result, check_scalar_result


@pytest.mark.gui
def test_image_fft_interactive():
    """2D FFT interactive test."""
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    from sigima_.tests import vistools

    with qt_app_context():
        # Create a 2D ring image
        execenv.print("Generating 2D ring image...", end=" ")
        data = ctd.create_ring_image(ctd.RingParam()).data
        execenv.print("OK")

        # FFT
        execenv.print("Computing FFT of image...", end=" ")
        f = alg.fft2d(data)
        data2 = alg.ifft2d(f)
        execenv.print("OK")
        execenv.print("Comparing original and FFT/iFFT images...", end=" ")
        check_array_result(
            "Image FFT/iFFT", np.array(data2.real, dtype=data.dtype), data, rtol=1e-3
        )
        execenv.print("OK")

        images = [data, f.real, f.imag, np.abs(f), data2.real, data2.imag]
        titles = ["Original", "Re(FFT)", "Im(FFT)", "Abs(FFT)", "Re(iFFT)", "Im(iFFT)"]
        vistools.view_images_side_by_side(images, titles, rows=2, title="2D FFT/iFFT")


@pytest.mark.validation
def test_image_zero_padding() -> None:
    """2D FFT zero padding validation test."""
    ima1 = ctd.create_checkerboard()
    rows, cols = 2, 2
    param = sigima_.param.ZeroPadding2DParam.create(rows=rows, cols=cols)
    assert param.strategy == "custom", (
        f"Wrong default strategy: {param.strategy} (expected 'custom')"
    )

    # Validate the zero padding with bottom-right position
    param.position = "bottom-right"
    ima2 = sigima_image.zero_padding(ima1, param)
    sh1, sh2 = ima1.data.shape, ima2.data.shape
    exp_sh2 = (sh1[0] + rows, sh1[1] + cols)
    execenv.print("Validating zero padding for bottom-right position...", end=" ")
    assert sh2 == exp_sh2, f"Wrong shape: {sh2} (expected {exp_sh2})"
    assert np.all(ima2.data[0 : sh1[0], 0 : sh1[1]] == ima1.data), (
        "Altered data in original image area"
    )
    assert np.all(ima2.data[sh1[0] : sh2[0], sh1[1] : sh2[1]] == 0), (
        "Altered data in padded area"
    )
    execenv.print("OK")

    # Validate the zero padding with center position
    param.position = "center"
    ima3 = sigima_image.zero_padding(ima1, param)
    sh3 = ima3.data.shape
    exp_sh3 = (sh1[0] + rows, sh1[1] + cols)
    execenv.print("Validating zero padding for center position...", end=" ")
    assert sh3 == exp_sh3, f"Wrong shape: {sh3} (expected {exp_sh3})"
    assert np.all(
        ima3.data[rows // 2 : sh1[0] + rows // 2, cols // 2 : sh1[1] + cols // 2]
        == ima1.data
    ), "Altered data in original image area"
    assert np.all(ima3.data[0 : rows // 2, :] == 0), "Altered data in padded area (top)"
    assert np.all(ima3.data[sh1[0] + rows // 2 :, :] == 0), (
        "Altered data in padded area (bottom)"
    )
    assert np.all(ima3.data[:, 0 : cols // 2] == 0), (
        "Altered data in padded area (left)"
    )
    assert np.all(ima3.data[:, sh1[1] + cols // 2 :] == 0), (
        "Altered data in padded area (right)"
    )
    execenv.print("OK")

    # Validate zero padding with strategies other than custom size
    # Image size is (200, 300) and the next power of 2 is (256, 512)
    # The multiple of 64 is (256, 320)
    ima4 = sigima_.obj.create_image("", np.zeros((200, 300)))
    for strategy, (exp_rows, exp_cols) in (
        ("next_pow2", (56, 212)),
        ("multiple_of_64", (56, 20)),
    ):
        param = sigima_.param.ZeroPadding2DParam.create(strategy=strategy)
        param.update_from_obj(ima4)
        assert param.rows == exp_rows, (
            f"Wrong row number for '{param.strategy}' strategy: {param.rows}"
            f" (expected {exp_rows})"
        )
        assert param.cols == exp_cols, (
            f"Wrong column number for '{param.strategy}' strategy: {param.cols}"
            f" (expected {exp_cols})"
        )


@pytest.mark.validation
def test_image_fft() -> None:
    """2D FFT validation test."""
    ima1 = ctd.create_checkerboard()
    fft = sigima_image.fft(ima1)
    ifft = sigima_image.ifft(fft)

    # Check that the inverse FFT reconstructs the original image
    check_array_result("Checkerboard image FFT/iFFT", ifft.data.real, ima1.data)

    # Parseval's Theorem Validation
    original_energy = np.sum(np.abs(ima1.data) ** 2)
    transformed_energy = np.sum(np.abs(fft.data) ** 2) / (ima1.data.size)
    check_scalar_result("Parseval's Theorem", transformed_energy, original_energy)


@pytest.mark.skip(reason="Already covered by the `test_image_fft` test.")
@pytest.mark.validation
def test_image_ifft() -> None:
    """2D iFFT validation test."""
    # This is just a way of marking the iFFT test as a validation test because it is
    # already covered by the FFT test above (there is no need to repeat the same test).


@pytest.mark.validation
def test_image_magnitude_spectrum() -> None:
    """2D magnitude spectrum validation test."""
    ima1 = ctd.create_checkerboard()
    fft = sigima_image.fft(ima1)
    mag = sigima_image.magnitude_spectrum(ima1)

    # Check that the magnitude spectrum is correct
    exp = np.abs(fft.data)
    check_array_result("Checkerboard image FFT magnitude spectrum", mag.data, exp)


@pytest.mark.validation
def test_image_phase_spectrum() -> None:
    """2D phase spectrum validation test."""
    ima1 = ctd.create_checkerboard()
    fft = sigima_image.fft(ima1)
    phase = sigima_image.phase_spectrum(ima1)

    # Check that the phase spectrum is correct
    exp = np.rad2deg(np.angle(fft.data))
    check_array_result("Checkerboard image FFT phase spectrum", phase.data, exp)


@pytest.mark.validation
def test_image_psd() -> None:
    """2D Power Spectral Density validation test."""
    ima1 = ctd.create_checkerboard()
    psd = sigima_image.psd(ima1)

    # Check that the PSD is correct
    exp = np.abs(sigima_image.fft(ima1).data) ** 2
    check_array_result("Checkerboard image PSD", psd.data, exp)


if __name__ == "__main__":
    test_image_fft_interactive()
    test_image_zero_padding()
    test_image_fft()
    test_image_magnitude_spectrum()
    test_image_phase_spectrum()
    test_image_psd()
