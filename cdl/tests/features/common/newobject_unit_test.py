# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
New signal/image test

Testing functions related to signal/image creation.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

from collections.abc import Generator

from guidata.qthelpers import qt_app_context

import sigima_.obj
from cdl.env import execenv
from cdl.gui.newobject import create_image_gui, create_signal_gui
from sigima_.tests.vistools import view_curves, view_images


def iterate_signal_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[sigima_.obj.SignalObj, None, None]:
    """Iterate over all possible signals created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over signal types (size={data_size}, non_zero={non_zero}):"
        )
    for stype in sigima_.obj.SignalTypes:
        if non_zero and stype in (sigima_.obj.SignalTypes.ZEROS,):
            continue
        if verbose:
            execenv.print(f"    {stype.value}")
        base_param = sigima_.obj.NewSignalParam.create(stype=stype, size=data_size)
        if stype == sigima_.obj.SignalTypes.UNIFORMRANDOM:
            extra_param = sigima_.obj.UniformRandomParam()
        elif stype == sigima_.obj.SignalTypes.NORMALRANDOM:
            extra_param = sigima_.obj.NormalRandomParam()
        elif stype in (
            sigima_.obj.SignalTypes.GAUSS,
            sigima_.obj.SignalTypes.LORENTZ,
            sigima_.obj.SignalTypes.VOIGT,
        ):
            extra_param = sigima_.obj.GaussLorentzVoigtParam()
        elif stype in (
            sigima_.obj.SignalTypes.SINUS,
            sigima_.obj.SignalTypes.COSINUS,
            sigima_.obj.SignalTypes.SAWTOOTH,
            sigima_.obj.SignalTypes.TRIANGLE,
            sigima_.obj.SignalTypes.SQUARE,
            sigima_.obj.SignalTypes.SINC,
        ):
            extra_param = sigima_.obj.PeriodicParam()
        elif stype == sigima_.obj.SignalTypes.STEP:
            extra_param = sigima_.obj.StepParam()
        elif stype == sigima_.obj.SignalTypes.EXPONENTIAL:
            extra_param = sigima_.obj.ExponentialParam()
        elif stype == sigima_.obj.SignalTypes.PULSE:
            extra_param = sigima_.obj.PulseParam()
        elif stype == sigima_.obj.SignalTypes.POLYNOMIAL:
            extra_param = sigima_.obj.PolyParam()
        elif stype == sigima_.obj.SignalTypes.EXPERIMENTAL:
            extra_param = sigima_.obj.ExperimentalSignalParam()
        else:
            extra_param = None
        signal = sigima_.obj.create_signal_from_param(
            base_param, extra_param=extra_param
        )
        if stype == sigima_.obj.SignalTypes.ZEROS:
            assert (signal.y == 0).all()
        yield signal


def iterate_image_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[sigima_.obj.ImageObj, None, None]:
    """Iterate over all possible images created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over image types (size={data_size}, non_zero={non_zero}):"
        )
    for itype in sigima_.obj.ImageTypes:
        if non_zero and itype in (
            sigima_.obj.ImageTypes.EMPTY,
            sigima_.obj.ImageTypes.ZEROS,
        ):
            continue
        if verbose:
            execenv.print(f"    {itype.value}")
        yield from _iterate_image_datatypes(itype, data_size, verbose)


def _iterate_image_datatypes(
    itype: sigima_.obj.ImageTypes, data_size: int, verbose: bool
) -> Generator[sigima_.obj.ImageObj | None, None, None]:
    for dtype in sigima_.obj.ImageDatatypes:
        if verbose:
            execenv.print(f"      {dtype.value}")
        base_param = sigima_.obj.NewImageParam.create(
            itype=itype, dtype=dtype, width=data_size, height=data_size
        )
        extra_param = _get_additional_param(itype, dtype)
        image = sigima_.obj.create_image_from_param(base_param, extra_param=extra_param)
        if image is not None:
            _test_image_data(itype, image)
        yield image


def _get_additional_param(
    itype: sigima_.obj.ImageTypes, dtype: sigima_.obj.ImageDatatypes
) -> (
    sigima_.obj.Gauss2DParam
    | sigima_.obj.UniformRandomParam
    | sigima_.obj.NormalRandomParam
    | None
):
    if itype == sigima_.obj.ImageTypes.GAUSS:
        addparam = sigima_.obj.Gauss2DParam()
        addparam.x0 = addparam.y0 = 3
        addparam.sigma = 5
    elif itype == sigima_.obj.ImageTypes.UNIFORMRANDOM:
        addparam = sigima_.obj.UniformRandomParam()
        addparam.set_from_datatype(dtype.value)
    elif itype == sigima_.obj.ImageTypes.NORMALRANDOM:
        addparam = sigima_.obj.NormalRandomParam()
        addparam.set_from_datatype(dtype.value)
    else:
        addparam = None
    return addparam


def _test_image_data(
    itype: sigima_.obj.ImageTypes, image: sigima_.obj.ImageObj
) -> None:
    """
    Tests the data of an image based on its type.

    Args:
        itype: The type of the image.
        image: The image object containing the data to be tested.

    Raises:
        AssertionError: If the image data does not match the expected values
         for the given image type.
    """
    if itype == sigima_.obj.ImageTypes.ZEROS:
        assert (image.data == 0).all()
    else:
        assert image.data is not None


def all_combinations_test() -> None:
    """Test all combinations for new signal/image feature"""
    execenv.print(f"Testing {all_combinations_test.__name__}:")
    execenv.print(f"  Signal types ({len(sigima_.obj.SignalTypes)}):")
    for signal in iterate_signal_creation():
        assert signal.x is not None and signal.y is not None
    execenv.print(f"  Image types ({len(sigima_.obj.ImageTypes)}):")
    for image in iterate_image_creation():
        assert image.data is not None
    execenv.print(f"{all_combinations_test.__name__} OK")


def __new_signal_test() -> None:
    """Test new signal feature"""
    edit = not execenv.unattended
    signal = create_signal_gui(None, edit=edit)
    if signal is not None:
        data = (signal.x, signal.y)
        view_curves([data], name=__new_signal_test.__name__, title=signal.title)


def __new_image_test() -> None:
    """Test new image feature"""
    # Test with no input parameter
    edit = not execenv.unattended
    image = create_image_gui(None, edit=edit)
    if image is not None:
        view_images(image.data, name=__new_image_test.__name__, title=image.title)
    # Test with parametered 2D-Gaussian
    base_param = sigima_.obj.NewImageParam.create(itype=sigima_.obj.ImageTypes.GAUSS)
    extra_param = sigima_.obj.Gauss2DParam()
    extra_param.x0 = extra_param.y0 = 3
    extra_param.sigma = 5
    image = create_image_gui(base_param, extra_param=extra_param, edit=edit)
    if image is not None:
        view_images(image.data, name=__new_image_test.__name__, title=image.title)


def test_new_object() -> None:
    """Test new signal/image feature"""
    all_combinations_test()
    with qt_app_context():
        __new_signal_test()
        __new_image_test()


if __name__ == "__main__":
    test_new_object()
