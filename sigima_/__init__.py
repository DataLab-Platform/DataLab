# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
sigima_
======

Scientific computing engine for 1D signals and 2D images,
part of the DataLab open-source platform.
"""

# TODO: Annotations: currently, the `items_to_json` and `json_to_items` functions rely
#       on PlotPy's serialization functions, which are not compatible with `sigima`
#       annotations model. So, we have to implement the functions to convert `sigima`
#       annotations to PlotPy items and vice versa: `items_to_sigima_annotations` and
#       `sigima_annotations_to_items`.
#       Here are the locations where the functions will be used:
#       - `BaseObjPlotPyAdapter.add_annotations_from_items`
#       - `BaseDataPanel.__separate_view_finished`
#       - `RemoteClient.add_annotations_from_items`

# TODO: Rename "sigima_" (temporary name until the package is fully migrated)
#       to "sigima_" when the migration is complete.
# TODO: Move functions below to a separate module?
# TODO: Remove all dependencies on `cdl` package on all modules
# TODO: Remove all references for `Conf` (no need for it in `sigima_` package?)
# TODO: Add local translations for the `sigima_` package
# TODO: Add `pytest` infrastructure for the `sigima_` package
# TODO: Implement a I/O plugin system similar to the `cdl.plugins` module
# TODO: Implement a computation plugin system similar to the `cdl.plugins` module
# TODO: Handle the NumPy minimum requirement to v1.21 to use advanced type hints?
# TODO: Should we keep `PREFIX` attribute in `BaseObj`?

# pylint:disable=unused-import
# flake8: noqa

from sigima_.io import (
    read_images,
    read_signals,
    write_image,
    write_signal,
    read_image,
    read_signal,
)
from sigima_.obj import (
    CircularROI,
    ExponentialParam,
    Gauss2DParam,
    GaussLorentzVoigtParam,
    ImageDatatypes,
    ImageROI,
    ImageObj,
    ImageTypes,
    NewImageParam,
    NewSignalParam,
    NormalRandomParam,
    PeriodicParam,
    PolygonalROI,
    RectangularROI,
    ResultProperties,
    ResultShape,
    ROI1DParam,
    ROI2DParam,
    SegmentROI,
    ShapeTypes,
    SignalObj,
    SignalROI,
    SignalTypes,
    StepParam,
    TypeObj,
    TypeROI,
    UniformRandomParam,
    create_image,
    create_image_from_param,
    create_image_roi,
    create_signal,
    create_signal_from_param,
    create_signal_roi,
)

__version__ = "0.0.1"
__docurl__ = __homeurl__ = "https://datalab-platform.com/"
__supporturl__ = "https://github.com/DataLab-Platform/sigima_/issues/new/choose"

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
