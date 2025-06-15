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

from cdl.env import execenv
from cdl.gui.newobject import create_image_gui, create_signal_gui
from cdl.utils.vistools import view_curves, view_images
from sigima_ import model


def iterate_signal_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[model.SignalObj, None, None]:
    """Iterate over all possible signals created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over signal types (size={data_size}, non_zero={non_zero}):"
        )
    for stype in model.SignalTypes:
        if non_zero and stype in (model.SignalTypes.ZEROS,):
            continue
        if verbose:
            execenv.print(f"    {stype.value}")
        base_param = model.NewSignalParam.create(stype=stype, size=data_size)
        if stype == model.SignalTypes.UNIFORMRANDOM:
            extra_param = model.UniformRandomParam()
        elif stype == model.SignalTypes.NORMALRANDOM:
            extra_param = model.NormalRandomParam()
        elif stype in (
            model.SignalTypes.GAUSS,
            model.SignalTypes.LORENTZ,
            model.SignalTypes.VOIGT,
        ):
            extra_param = model.GaussLorentzVoigtParam()
        elif stype in (
            model.SignalTypes.SINUS,
            model.SignalTypes.COSINUS,
            model.SignalTypes.SAWTOOTH,
            model.SignalTypes.TRIANGLE,
            model.SignalTypes.SQUARE,
            model.SignalTypes.SINC,
        ):
            extra_param = model.PeriodicParam()
        elif stype == model.SignalTypes.STEP:
            extra_param = model.StepParam()
        elif stype == model.SignalTypes.EXPONENTIAL:
            extra_param = model.ExponentialParam()
        elif stype == model.SignalTypes.PULSE:
            extra_param = model.PulseParam()
        elif stype == model.SignalTypes.POLYNOMIAL:
            extra_param = model.PolyParam()
        elif stype == model.SignalTypes.EXPERIMENTAL:
            extra_param = model.ExperimentalSignalParam()
        else:
            extra_param = None
        signal = model.create_signal_from_param(base_param, extra_param=extra_param)
        if stype == model.SignalTypes.ZEROS:
            assert (signal.y == 0).all()
        yield signal


def iterate_image_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[model.ImageObj, None, None]:
    """Iterate over all possible images created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over image types (size={data_size}, non_zero={non_zero}):"
        )
    for itype in model.ImageTypes:
        if non_zero and itype in (model.ImageTypes.EMPTY, model.ImageTypes.ZEROS):
            continue
        if verbose:
            execenv.print(f"    {itype.value}")
        yield from _iterate_image_datatypes(itype, data_size, verbose)


def _iterate_image_datatypes(
    itype: model.ImageTypes, data_size: int, verbose: bool
) -> Generator[model.ImageObj | None, None, None]:
    for dtype in model.ImageDatatypes:
        if verbose:
            execenv.print(f"      {dtype.value}")
        base_param = model.NewImageParam.create(
            itype=itype, dtype=dtype, width=data_size, height=data_size
        )
        extra_param = _get_additional_param(itype, dtype)
        image = model.create_image_from_param(base_param, extra_param=extra_param)
        if image is not None:
            _test_image_data(itype, image)
        yield image


def _get_additional_param(
    itype: model.ImageTypes, dtype: model.ImageDatatypes
) -> model.Gauss2DParam | model.UniformRandomParam | model.NormalRandomParam | None:
    if itype == model.ImageTypes.GAUSS:
        addparam = model.Gauss2DParam()
        addparam.x0 = addparam.y0 = 3
        addparam.sigma = 5
    elif itype == model.ImageTypes.UNIFORMRANDOM:
        addparam = model.UniformRandomParam()
        addparam.set_from_datatype(dtype.value)
    elif itype == model.ImageTypes.NORMALRANDOM:
        addparam = model.NormalRandomParam()
        addparam.set_from_datatype(dtype.value)
    else:
        addparam = None
    return addparam


def _test_image_data(itype: model.ImageTypes, image: model.ImageObj) -> None:
    """
    Tests the data of an image based on its type.

    Args:
        itype: The type of the image.
        image: The image object containing the data to be tested.

    Raises:
        AssertionError: If the image data does not match the expected values
         for the given image type.
    """
    if itype == model.ImageTypes.ZEROS:
        assert (image.data == 0).all()
    else:
        assert image.data is not None


def all_combinations_test():
    """Test all combinations for new signal/image feature"""
    execenv.print(f"Testing {all_combinations_test.__name__}:")
    execenv.print(f"  Signal types ({len(model.SignalTypes)}):")
    for signal in iterate_signal_creation():
        assert signal.x is not None and signal.y is not None
    execenv.print(f"  Image types ({len(model.ImageTypes)}):")
    for image in iterate_image_creation():
        assert image.data is not None
    execenv.print(f"{all_combinations_test.__name__} OK")


def __new_signal_test():
    """Test new signal feature"""
    edit = not execenv.unattended
    signal = create_signal_gui(None, edit=edit)
    if signal is not None:
        data = (signal.x, signal.y)
        view_curves([data], name=__new_signal_test.__name__, title=signal.title)


def __new_image_test():
    """Test new image feature"""
    # Test with no input parameter
    edit = not execenv.unattended
    image = create_image_gui(None, edit=edit)
    if image is not None:
        view_images(image.data, name=__new_image_test.__name__, title=image.title)
    # Test with parametered 2D-Gaussian
    base_param = model.NewImageParam.create(itype=model.ImageTypes.GAUSS)
    extra_param = model.Gauss2DParam()
    extra_param.x0 = extra_param.y0 = 3
    extra_param.sigma = 5
    image = create_image_gui(base_param, extra_param=extra_param, edit=edit)
    if image is not None:
        view_images(image.data, name=__new_image_test.__name__, title=image.title)


def test_new_object():
    """Test new signal/image feature"""
    all_combinations_test()
    with qt_app_context():
        __new_signal_test()
        __new_image_test()


if __name__ == "__main__":
    # test_new_object()
    print(model.SignalTypes.get_choices())
