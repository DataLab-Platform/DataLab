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

import sigima_.obj as so
from cdl.env import execenv
from cdl.gui.newobject import create_image_gui, create_signal_gui
from cdl.utils.vistools import view_curves, view_images


def iterate_signal_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[so.SignalObj, None, None]:
    """Iterate over all possible signals created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over signal types (size={data_size}, non_zero={non_zero}):"
        )
    for stype in so.SignalTypes:
        if non_zero and stype in (so.SignalTypes.ZEROS,):
            continue
        if verbose:
            execenv.print(f"    {stype.value}")
        base_param = so.NewSignalParam.create(stype=stype, size=data_size)
        if stype == so.SignalTypes.UNIFORMRANDOM:
            extra_param = so.UniformRandomParam()
        elif stype == so.SignalTypes.NORMALRANDOM:
            extra_param = so.NormalRandomParam()
        elif stype in (
            so.SignalTypes.GAUSS,
            so.SignalTypes.LORENTZ,
            so.SignalTypes.VOIGT,
        ):
            extra_param = so.GaussLorentzVoigtParam()
        elif stype in (
            so.SignalTypes.SINUS,
            so.SignalTypes.COSINUS,
            so.SignalTypes.SAWTOOTH,
            so.SignalTypes.TRIANGLE,
            so.SignalTypes.SQUARE,
            so.SignalTypes.SINC,
        ):
            extra_param = so.PeriodicParam()
        elif stype == so.SignalTypes.STEP:
            extra_param = so.StepParam()
        elif stype == so.SignalTypes.EXPONENTIAL:
            extra_param = so.ExponentialParam()
        elif stype == so.SignalTypes.PULSE:
            extra_param = so.PulseParam()
        elif stype == so.SignalTypes.POLYNOMIAL:
            extra_param = so.PolyParam()
        elif stype == so.SignalTypes.EXPERIMENTAL:
            extra_param = so.ExperimentalSignalParam()
        else:
            extra_param = None
        signal = so.create_signal_from_param(base_param, extra_param=extra_param)
        if stype == so.SignalTypes.ZEROS:
            assert (signal.y == 0).all()
        yield signal


def iterate_image_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[so.ImageObj, None, None]:
    """Iterate over all possible images created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over image types (size={data_size}, non_zero={non_zero}):"
        )
    for itype in so.ImageTypes:
        if non_zero and itype in (so.ImageTypes.EMPTY, so.ImageTypes.ZEROS):
            continue
        if verbose:
            execenv.print(f"    {itype.value}")
        yield from _iterate_image_datatypes(itype, data_size, verbose)


def _iterate_image_datatypes(
    itype: so.ImageTypes, data_size: int, verbose: bool
) -> Generator[so.ImageObj | None, None, None]:
    for dtype in so.ImageDatatypes:
        if verbose:
            execenv.print(f"      {dtype.value}")
        base_param = so.NewImageParam.create(
            itype=itype, dtype=dtype, width=data_size, height=data_size
        )
        extra_param = _get_additional_param(itype, dtype)
        image = so.create_image_from_param(base_param, extra_param=extra_param)
        if image is not None:
            _test_image_data(itype, image)
        yield image


def _get_additional_param(
    itype: so.ImageTypes, dtype: so.ImageDatatypes
) -> so.Gauss2DParam | so.UniformRandomParam | so.NormalRandomParam | None:
    if itype == so.ImageTypes.GAUSS:
        addparam = so.Gauss2DParam()
        addparam.x0 = addparam.y0 = 3
        addparam.sigma = 5
    elif itype == so.ImageTypes.UNIFORMRANDOM:
        addparam = so.UniformRandomParam()
        addparam.set_from_datatype(dtype.value)
    elif itype == so.ImageTypes.NORMALRANDOM:
        addparam = so.NormalRandomParam()
        addparam.set_from_datatype(dtype.value)
    else:
        addparam = None
    return addparam


def _test_image_data(itype: so.ImageTypes, image: so.ImageObj) -> None:
    """
    Tests the data of an image based on its type.

    Args:
        itype: The type of the image.
        image: The image object containing the data to be tested.

    Raises:
        AssertionError: If the image data does not match the expected values
         for the given image type.
    """
    if itype == so.ImageTypes.ZEROS:
        assert (image.data == 0).all()
    else:
        assert image.data is not None


def all_combinations_test() -> None:
    """Test all combinations for new signal/image feature"""
    execenv.print(f"Testing {all_combinations_test.__name__}:")
    execenv.print(f"  Signal types ({len(so.SignalTypes)}):")
    for signal in iterate_signal_creation():
        assert signal.x is not None and signal.y is not None
    execenv.print(f"  Image types ({len(so.ImageTypes)}):")
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
    base_param = so.NewImageParam.create(itype=so.ImageTypes.GAUSS)
    extra_param = so.Gauss2DParam()
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
