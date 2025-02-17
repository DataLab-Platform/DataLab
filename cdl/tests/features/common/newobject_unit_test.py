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
from cdl.obj import (
    Gauss2DParam,
    ImageDatatypes,
    ImageObj,
    ImageTypes,
    NormalRandomParam,
    SignalObj,
    SignalTypes,
    UniformRandomParam,
    create_image_from_param,
    create_signal_from_param,
    new_image_param,
    new_signal_param,
)
from cdl.utils.vistools import view_curves, view_images


def iterate_signal_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[SignalObj, None, None]:
    """Iterate over all possible signals created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over signal types (size={data_size}, non_zero={non_zero}):"
        )
    for stype in SignalTypes:
        if non_zero and stype in (SignalTypes.ZEROS,):
            continue
        if verbose:
            execenv.print(f"    {stype.value}")
        newparam = new_signal_param(stype=stype, size=data_size)
        if stype == SignalTypes.UNIFORMRANDOM:
            addparam = UniformRandomParam()
        elif stype == SignalTypes.NORMALRANDOM:
            addparam = NormalRandomParam()
        else:
            addparam = None
        signal = create_signal_from_param(newparam, addparam=addparam)
        if stype == SignalTypes.ZEROS:
            assert (signal.y == 0).all()
        yield signal


def iterate_image_creation(
    data_size: int = 500, non_zero: bool = False, verbose: bool = True
) -> Generator[ImageObj, None, None]:
    """Iterate over all possible images created from parameters"""
    if verbose:
        execenv.print(
            f"  Iterating over image types (size={data_size}, non_zero={non_zero}):"
        )
    for itype in ImageTypes:
        if non_zero and itype in (ImageTypes.EMPTY, ImageTypes.ZEROS):
            continue
        if verbose:
            execenv.print(f"    {itype.value}")
        yield from _iterate_image_datatypes(itype, data_size, verbose)


def _iterate_image_datatypes(
    itype: ImageTypes, data_size: int, verbose: bool
) -> Generator[ImageObj | None, None, None]:
    for dtype in ImageDatatypes:
        if verbose:
            execenv.print(f"      {dtype.value}")
        newparam = new_image_param(
            itype=itype, dtype=dtype, width=data_size, height=data_size
        )
        addparam = _get_additional_param(itype, dtype)
        image = create_image_from_param(newparam, addparam=addparam)
        if image is not None:
            _test_image_data(itype, image)
        yield image


def _get_additional_param(
    itype: ImageTypes, dtype: ImageDatatypes
) -> Gauss2DParam | UniformRandomParam | NormalRandomParam | None:
    if itype == ImageTypes.GAUSS:
        addparam = Gauss2DParam()
        addparam.x0 = addparam.y0 = 3
        addparam.sigma = 5
    elif itype == ImageTypes.UNIFORMRANDOM:
        addparam = UniformRandomParam()
        addparam.set_from_datatype(dtype.value)
    elif itype == ImageTypes.NORMALRANDOM:
        addparam = NormalRandomParam()
        addparam.set_from_datatype(dtype.value)
    else:
        addparam = None
    return addparam


def _test_image_data(itype: ImageTypes, image: ImageObj) -> None:
    """
    Tests the data of an image based on its type.

    Args:
    itype (ImageTypes): The type of the image.
    image (Image): The image object containing the data to be tested.

    Raises:
    AssertionError: If the image data does not match the expected values for the given image type.
    """
    if itype == ImageTypes.ZEROS:
        assert (image.data == 0).all()
    else:
        assert image.data is not None


def all_combinations_test():
    """Test all combinations for new signal/image feature"""
    execenv.print(f"Testing {all_combinations_test.__name__}:")
    execenv.print(f"  Signal types ({len(SignalTypes)}):")
    for signal in iterate_signal_creation():
        assert signal.x is not None and signal.y is not None
    execenv.print(f"  Image types ({len(ImageTypes)}):")
    for image in iterate_image_creation():
        assert image.data is not None
    execenv.print(f"{all_combinations_test.__name__} OK")


def __new_signal_test():
    """Test new signal feature"""
    edit = not execenv.unattended
    signal = create_signal_from_param(None, edit=edit)
    if signal is not None:
        data = (signal.x, signal.y)
        view_curves([data], name=__new_signal_test.__name__, title=signal.title)


def __new_image_test():
    """Test new image feature"""
    # Test with no input parameter
    edit = not execenv.unattended
    image = create_image_from_param(None, edit=edit)
    if image is not None:
        view_images(image.data, name=__new_image_test.__name__, title=image.title)
    # Test with parametered 2D-Gaussian
    newparam = new_image_param(itype=ImageTypes.GAUSS)
    addparam = Gauss2DParam()
    addparam.x0 = addparam.y0 = 3
    addparam.sigma = 5
    image = create_image_from_param(newparam, addparam=addparam, edit=edit)
    if image is not None:
        view_images(image.data, name=__new_image_test.__name__, title=image.title)


def test_new_object():
    """Test new signal/image feature"""
    all_combinations_test()
    with qt_app_context():
        __new_signal_test()
        __new_image_test()


if __name__ == "__main__":
    test_new_object()
