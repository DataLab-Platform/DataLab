# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image processor object (see parent package :mod:`datalab.gui.processor`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import importlib
from typing import Literal

import numpy as np
import sigima.params
import sigima.proc.base as sigima_base
import sigima.proc.image as sigima_image
from guidata.qthelpers import exec_dialog
from plotpy.widgets.resizedialog import ResizeDialog
from qtpy import QtWidgets as QW
from sigima.objects import (
    ImageObj,
    ImageROI,
    NormalDistributionParam,
    PoissonDistributionParam,
    ROI2DParam,
    UniformDistributionParam,
)
from sigima.objects.scalar import GeometryResult
from sigima.proc.decorator import ComputationMetadata
from sigima.proc.transformations import transformer

from datalab.adapters_metadata import GeometryAdapter
from datalab.config import APP_NAME, _
from datalab.gui.processor.base import BaseProcessor
from datalab.gui.profiledialog import ProfileExtractionDialog
from datalab.gui.roigrideditor import ImageGridROIEditor
from datalab.objectmodel import get_uuid
from datalab.utils.qthelpers import create_progress_bar, qt_try_except
from datalab.widgets import imagebackground


def apply_geometry_transform(
    obj: ImageObj,
    operation: Literal[
        "translate", "scale", "rotate90", "rotate270", "fliph", "flipv", "transpose"
    ],
    **kwargs,
) -> None:
    """Apply a geometric transformation to all geometry results in an object.

    This uses the Sigima transformation system for proper geometric operations.
    For image objects, rotations are performed around the image center to match
    how the image data is transformed.

    Args:
        obj: The object containing geometry results to transform
        operation: The transformation operation name
        **kwargs: Optional parameters for the transformation (e.g., angle for rotate)
    """
    assert operation in [
        "translate",
        "scale",
        "rotate90",
        "rotate270",
        "fliph",
        "flipv",
        "transpose",
    ], f"Unknown operation: {operation}"
    if operation == "translate":
        if not kwargs or "dx" not in kwargs or "dy" not in kwargs:
            raise ValueError("translate operation requires 'dx' and 'dy' parameters")
        dx, dy = kwargs["dx"], kwargs["dy"]
    elif operation == "scale":
        if not kwargs or "sx" not in kwargs or "sy" not in kwargs:
            raise ValueError("scale operation requires 'sx' and 'sy' parameters")
        sx, sy = kwargs["sx"], kwargs["sy"]
    for adapter in list(GeometryAdapter.iterate_from_obj(obj)):
        geometry = adapter.result
        assert geometry is not None, "Geometry should not be None"
        assert len(geometry.coords) > 0, "Geometry coordinates should not be empty"
        if operation == "translate":
            tr_geometry = transformer.translate(geometry, dx, dy)
        elif operation == "scale":
            tr_geometry = transformer.scale(geometry, sx, sy, (obj.xc, obj.yc))
        elif operation == "rotate90":
            tr_geometry = transformer.rotate(geometry, -np.pi / 2, (obj.xc, obj.yc))
        elif operation == "rotate270":
            tr_geometry = transformer.rotate(geometry, np.pi / 2, (obj.xc, obj.yc))
        elif operation == "fliph":
            tr_geometry = transformer.fliph(geometry, obj.xc)
        elif operation == "flipv":
            tr_geometry = transformer.flipv(geometry, obj.yc)
        elif operation == "transpose":
            tr_geometry = transformer.transpose(geometry)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Remove the old geometry and add the transformed one
        adapter.remove_from(obj)
        tr_adapter = GeometryAdapter(tr_geometry)
        tr_adapter.add_to(obj)


class GeometricTransformWrapper:
    """Pickleable wrapper for geometric transformation functions.

    This class creates a callable wrapper that can be pickled, unlike nested functions.
    Instead of storing the function directly, it stores the module path and function
    name to allow proper pickling.
    """

    def __init__(self, func, operation: str):
        self.operation = operation

        # Store function reference for execution
        self.func = func

        # Store function module and name for pickling
        self.__module__ = func.__module__
        self.__qualname__ = func.__qualname__
        self.__annotations__ = getattr(func, "__annotations__", {})
        self.__name__ = getattr(func, "__name__", str(func))

        # Copy the __wrapped__ attribute if it exists (for Sigima compatibility)
        # Note: We don't copy __wrapped__ as it may contain unpickleable references
        # The wrapper functionality will still work without it

        # Copy Sigima computation metadata (required for validation)
        computation_metadata_attr = "__computation_function_metadata"
        if hasattr(func, computation_metadata_attr):
            setattr(
                self,
                computation_metadata_attr,
                getattr(func, computation_metadata_attr),
            )

    def __call__(self, src_obj, param=None):
        """Call the wrapped function and apply geometry transformations."""
        # Call the original function
        if param is not None:
            dst_obj = self.func(src_obj, param)
        else:
            dst_obj = self.func(src_obj)
        apply_geometry_transform(dst_obj, operation=self.operation)
        return dst_obj

    def __getstate__(self):
        """Custom pickling: exclude the function reference."""
        # Build state manually to avoid any problematic attributes
        state = {
            "operation": self.operation,
            "__module__": self.__module__,
            "__qualname__": self.__qualname__,
            "__annotations__": self.__annotations__,
            "__name__": self.__name__,
        }

        # Store function information for reconstruction
        if hasattr(self, "func"):
            state["_func_module"] = self.func.__module__
            state["_func_name"] = self.func.__name__

        # Note: We don't copy __wrapped__ as it may contain unpickleable references

        # Copy computation metadata safely
        computation_metadata_attr = "__computation_function_metadata"
        if hasattr(self, computation_metadata_attr):
            metadata = getattr(self, computation_metadata_attr)
            # Store as a dict to avoid any pickling issues with the object itself
            if hasattr(metadata, "__dict__"):
                state[computation_metadata_attr] = metadata.__dict__.copy()
            else:
                state[computation_metadata_attr] = metadata

        return state

    def __setstate__(self, state):
        """Custom unpickling: restore the function reference."""
        self.__dict__.update(state)
        # Restore function from module and name
        if "_func_module" in state and "_func_name" in state:
            module = importlib.import_module(state["_func_module"])
            self.func = getattr(module, state["_func_name"])

        # Reconstruct computation metadata if it was stored as dict
        computation_metadata_attr = "__computation_function_metadata"
        if computation_metadata_attr in state:
            metadata_dict = state[computation_metadata_attr]
            if isinstance(metadata_dict, dict):
                metadata = ComputationMetadata(**metadata_dict)
                setattr(self, computation_metadata_attr, metadata)


class ImageProcessor(BaseProcessor[ImageROI, ROI2DParam]):
    """Object handling image processing: operations, processing, analysis"""

    # pylint: disable=duplicate-code

    def _wrap_geometric_transform(self, func, operation: str):
        """Wrap a geometric transformation function to apply geometry transforms.

        Args:
            func: The original Sigima function
            operation: The operation name for geometry transformation

        Returns:
            Pickleable wrapped function that applies geometry transformations
        """
        return GeometricTransformWrapper(func, operation)

    def register_operations(self) -> None:
        """Register operations."""
        self.register_n_to_1(sigima_image.addition, _("Sum"), icon_name="sum.svg")
        self.register_n_to_1(
            sigima_image.average, _("Average"), icon_name="average.svg"
        )
        self.register_n_to_1(
            sigima_image.standard_deviation,
            _("Standard deviation"),
            icon_name="std.svg",
        )
        self.register_2_to_1(
            sigima_image.difference,
            _("Difference"),
            icon_name="difference.svg",
            obj2_name=_("image to subtract"),
        )
        self.register_2_to_1(
            sigima_image.quadratic_difference,
            _("Quadratic Difference"),
            icon_name="quadratic_difference.svg",
            obj2_name=_("image to subtract"),
        )
        self.register_n_to_1(
            sigima_image.product, _("Product"), icon_name="product.svg"
        )
        self.register_2_to_1(
            sigima_image.division,
            _("Division"),
            icon_name="division.svg",
            obj2_name=_("divider"),
        )
        self.register_1_to_1(
            sigima_image.inverse, _("Inverse"), icon_name="inverse.svg"
        )
        self.register_2_to_1(
            sigima_image.arithmetic,
            _("Arithmetic"),
            paramclass=sigima_base.ArithmeticParam,
            icon_name="arithmetic.svg",
            obj2_name=_("signal to operate with"),
        )
        self.register_1_to_1(
            sigima_image.addition_constant,
            _("Add constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_add.svg",
        )
        self.register_1_to_1(
            sigima_image.difference_constant,
            _("Subtract constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_subtract.svg",
        )
        self.register_1_to_1(
            sigima_image.product_constant,
            _("Multiply by constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_multiply.svg",
        )
        self.register_1_to_1(
            sigima_image.division_constant,
            _("Divide by constant"),
            paramclass=sigima_base.ConstantParam,
            icon_name="constant_divide.svg",
        )
        self.register_1_to_1(
            sigima_image.absolute, _("Absolute value"), icon_name="abs.svg"
        )
        self.register_1_to_1(
            sigima_image.phase,
            _("Phase"),
            paramclass=sigima.params.PhaseParam,
            icon_name="phase.svg",
        )
        self.register_2_to_1(
            sigima_image.complex_from_magnitude_phase,
            _("Combine with phase"),
            paramclass=sigima_base.AngleUnitParam,
            icon_name="complex_from_magnitude_phase.svg",
            comment=_("Create a complex-valued image from magnitude and phase"),
            obj2_name=_("phase"),
        )
        self.register_1_to_1(sigima_image.real, _("Real part"), icon_name="re.svg")
        self.register_1_to_1(sigima_image.imag, _("Imaginary part"), icon_name="im.svg")
        self.register_2_to_1(
            sigima_image.complex_from_real_imag,
            _("Combine with imaginary part"),
            icon_name="complex_from_real_imag.svg",
            comment=_("Create a complex-valued image from real and imaginary parts"),
            obj2_name=_("imaginary part"),
        )
        self.register_1_to_1(
            sigima_image.astype,
            _("Convert data type"),
            paramclass=sigima.params.DataTypeIParam,
            icon_name="convert_dtype.svg",
        )
        self.register_1_to_1(sigima_image.exp, _("Exponential"), icon_name="exp.svg")
        self.register_1_to_1(
            sigima_image.log10, _("Logarithm (base 10)"), icon_name="log10.svg"
        )
        self.register_1_to_1(sigima_image.logp1, "Log10(z+n)")
        self.register_2_to_1(
            sigima_image.flatfield,
            _("Flat-field correction"),
            sigima_image.FlatFieldParam,
            obj2_name=_("flat field image"),
        )
        # Flip or rotation
        self.register_1_to_1(
            self._wrap_geometric_transform(sigima_image.fliph, "fliph"),
            _("Flip horizontally"),
            icon_name="flip_horizontally.svg",
        )
        self.register_1_to_1(
            self._wrap_geometric_transform(sigima_image.transpose, "transpose"),
            _("Flip diagonally"),
            icon_name="swap_x_y.svg",
        )
        self.register_1_to_1(
            self._wrap_geometric_transform(sigima_image.flipv, "flipv"),
            _("Flip vertically"),
            icon_name="flip_vertically.svg",
        )
        self.register_1_to_1(
            self._wrap_geometric_transform(sigima_image.rotate270, "rotate270"),
            _("Rotate %s right") % "90°",
            icon_name="rotate_right.svg",
        )
        self.register_1_to_1(
            self._wrap_geometric_transform(sigima_image.rotate90, "rotate90"),
            _("Rotate %s left") % "90°",
            icon_name="rotate_left.svg",
        )
        self.register_1_to_1(
            sigima_image.rotate, _("Rotate by..."), sigima_image.RotateParam
        )
        # Intensity profiles
        self.register_1_to_1(
            sigima_image.line_profile,
            _("Line profile"),
            sigima_image.LineProfileParam,
            icon_name="profile.svg",
            edit=False,
        )
        self.register_1_to_1(
            sigima_image.segment_profile,
            _("Segment profile"),
            sigima_image.SegmentProfileParam,
            icon_name="profile_segment.svg",
            edit=False,
        )
        self.register_1_to_1(
            sigima_image.average_profile,
            _("Average profile"),
            sigima_image.AverageProfileParam,
            icon_name="profile_average.svg",
            edit=False,
        )
        self.register_1_to_1(
            sigima_image.radial_profile,
            _("Radial profile"),
            sigima_image.RadialProfileParam,
            icon_name="profile_radial.svg",
        )
        self.register_2_to_1(
            sigima_image.convolution,
            _("Convolution"),
            icon_name="convolution.svg",
            obj2_name=_("kernel to convolve with"),
        )
        self.register_2_to_1(
            sigima_image.deconvolution,
            _("Deconvolution"),
            icon_name="deconvolution.svg",
            obj2_name=_("kernel to deconvolve with"),
        )

    def register_processing(self) -> None:
        """Register processing functions."""
        # Axis transformation
        self.register_1_to_1(
            sigima_image.calibration,
            _("Linear calibration"),
            sigima_image.XYZCalibrateParam,
            comment=_(
                "Apply linear calibration to the X, Y or Z axis:\n"
                "  • x' = ax + b\n"
                "  • y' = ay + b\n"
                "  • z' = az + b"
            ),
        )
        self.register_1_to_1(
            sigima_image.transpose,
            _("Swap X/Y axes"),
            icon_name="swap_x_y.svg",
        )
        # Level adjustment
        self.register_1_to_1(
            sigima_image.normalize,
            _("Normalize"),
            paramclass=sigima_base.NormalizeParam,
            icon_name="normalize.svg",
        )
        self.register_1_to_1(
            sigima_image.clip, _("Clipping"), sigima_base.ClipParam, "clip.svg"
        )
        self.register_1_to_1(
            sigima_image.offset_correction,
            _("Offset correction"),
            ROI2DParam,
            comment=_("Evaluate and subtract the offset value from the data"),
            icon_name="offset_correction.svg",
        )
        # Noise addition
        self.register_1_to_1(
            sigima_image.add_gaussian_noise,
            _("Add Gaussian noise"),
            NormalDistributionParam,
        )
        self.register_1_to_1(
            sigima_image.add_poisson_noise,
            _("Add Poisson noise"),
            PoissonDistributionParam,
        )
        self.register_1_to_1(
            sigima_image.add_uniform_noise,
            _("Add uniform noise"),
            UniformDistributionParam,
        )
        # Noise reduction
        self.register_1_to_1(
            sigima_image.gaussian_filter,
            _("Gaussian filter"),
            sigima_base.GaussianParam,
        )
        self.register_1_to_1(
            sigima_image.moving_average,
            _("Moving average"),
            sigima_base.MovingAverageParam,
        )
        self.register_1_to_1(
            sigima_image.moving_median,
            _("Moving median"),
            sigima_base.MovingMedianParam,
        )
        self.register_1_to_1(sigima_image.wiener, _("Wiener filter"))
        self.register_1_to_1(
            sigima_image.erase,
            _("Erase area"),
            ROI2DParam,
            comment=_("Erase area in the image as defined by a region of interest"),
            icon_name="erase.svg",
        )
        # Fourier analysis
        self.register_1_to_1(
            sigima_image.zero_padding,
            _("Zero padding"),
            sigima_image.ZeroPadding2DParam,
            comment=_(
                "Zero padding is used to increase the frequency resolution of the FFT"
            ),
        )
        self.register_1_to_1(
            sigima_image.fft,
            _("FFT"),
            sigima_base.FFTParam,
            comment=_(
                "Fast Fourier Transform (FFT) is an estimation of the "
                "Discrete Fourier Transform (DFT). "
                "Results are complex numbers, but only the real part is plotted."
            ),
            edit=False,
        )
        self.register_1_to_1(
            sigima_image.ifft,
            _("Inverse FFT"),
            sigima_base.FFTParam,
            comment=_(
                "Inverse Fast Fourier Transform (IFFT) is an estimation of the "
                "Inverse Discrete Fourier Transform (IDFT). "
                "Results are complex numbers, but only the real part is plotted."
            ),
            edit=False,
        )
        self.register_1_to_1(
            sigima_image.magnitude_spectrum,
            _("Magnitude spectrum"),
            paramclass=sigima.params.SpectrumParam,
            comment=_(
                "Magnitude spectrum is the absolute value of the FFT result. "
                "It is a measure of the amplitude of the frequency components."
            ),
        )
        self.register_1_to_1(
            sigima_image.phase_spectrum,
            _("Phase spectrum"),
            comment=_(
                "Phase spectrum is the angle of the FFT result. "
                "It is a measure of the phase of the frequency components."
            ),
        )
        self.register_1_to_1(
            sigima_image.psd,
            _("Power spectral density"),
            paramclass=sigima.params.SpectrumParam,
            comment=_(
                "Power spectral density (PSD) is the square of the magnitude spectrum. "
                "It is a measure of the power of the frequency components."
            ),
        )
        # Frequency filters
        self.register_1_to_1(
            sigima_image.butterworth,
            _("Butterworth"),
            sigima_image.ButterworthParam,
        )
        self.register_1_to_1(
            sigima_image.gaussian_freq_filter,
            _("Gaussian bandpass"),
            sigima_image.GaussianFreqFilterParam,
        )
        # Thresholding
        self.register_1_to_1(
            sigima_image.threshold,
            _("Parametric thresholding"),
            sigima_image.ThresholdParam,
            comment=_(
                "Parametric thresholding allows to select a thresholding method "
                "and a threshold value."
            ),
        )
        self.register_1_to_1(sigima_image.threshold_isodata, _("ISODATA thresholding"))
        self.register_1_to_1(sigima_image.threshold_li, _("Li thresholding"))
        self.register_1_to_1(sigima_image.threshold_mean, _("Mean thresholding"))
        self.register_1_to_1(sigima_image.threshold_minimum, _("Minimum thresholding"))
        self.register_1_to_1(sigima_image.threshold_otsu, _("Otsu thresholding"))
        self.register_1_to_1(
            sigima_image.threshold_triangle, _("Triangle thresholding")
        )
        self.register_1_to_1(sigima_image.threshold_yen, _("Li thresholding"))
        # Exposure
        self.register_1_to_1(
            sigima_image.adjust_gamma,
            _("Gamma correction"),
            sigima_image.AdjustGammaParam,
        )
        self.register_1_to_1(
            sigima_image.adjust_log,
            _("Logarithmic correction"),
            sigima_image.AdjustLogParam,
        )
        self.register_1_to_1(
            sigima_image.adjust_sigmoid,
            _("Sigmoid correction"),
            sigima_image.AdjustSigmoidParam,
        )
        self.register_1_to_1(
            sigima_image.equalize_hist,
            _("Histogram equalization"),
            sigima_image.EqualizeHistParam,
        )
        self.register_1_to_1(
            sigima_image.equalize_adapthist,
            _("Adaptive histogram equalization"),
            sigima_image.EqualizeAdaptHistParam,
        )
        self.register_1_to_1(
            sigima_image.rescale_intensity,
            _("Intensity rescaling"),
            sigima_image.RescaleIntensityParam,
        )
        # Restoration
        self.register_1_to_1(
            sigima_image.denoise_tv,
            _("Total variation denoising"),
            sigima_image.DenoiseTVParam,
        )
        self.register_1_to_1(
            sigima_image.denoise_bilateral,
            _("Bilateral filter denoising"),
            sigima_image.DenoiseBilateralParam,
        )
        self.register_1_to_1(
            sigima_image.denoise_wavelet,
            _("Wavelet denoising"),
            sigima_image.DenoiseWaveletParam,
        )
        self.register_1_to_1(
            sigima_image.denoise_tophat,
            _("White Top-hat denoising"),
            sigima_image.MorphologyParam,
        )
        # Morphology
        self.register_1_to_1(
            sigima_image.white_tophat,
            _("White Top-Hat (disk)"),
            sigima_image.MorphologyParam,
        )
        self.register_1_to_1(
            sigima_image.black_tophat,
            _("Black Top-Hat (disk)"),
            sigima_image.MorphologyParam,
        )
        self.register_1_to_1(
            sigima_image.erosion,
            _("Erosion (disk)"),
            sigima_image.MorphologyParam,
        )
        self.register_1_to_1(
            sigima_image.dilation,
            _("Dilation (disk)"),
            sigima_image.MorphologyParam,
        )
        self.register_1_to_1(
            sigima_image.opening,
            _("Opening (disk)"),
            sigima_image.MorphologyParam,
        )
        self.register_1_to_1(
            sigima_image.closing,
            _("Closing (disk)"),
            sigima_image.MorphologyParam,
        )
        # Edge detection
        self.register_1_to_1(
            sigima_image.canny, _("Canny filter"), sigima_image.CannyParam
        )
        self.register_1_to_1(sigima_image.farid, _("Farid filter"))
        self.register_1_to_1(sigima_image.farid_h, _("Farid filter (horizontal)"))
        self.register_1_to_1(sigima_image.farid_v, _("Farid filter (vertical)"))
        self.register_1_to_1(sigima_image.laplace, _("Laplace filter"))
        self.register_1_to_1(sigima_image.prewitt, _("Prewitt filter"))
        self.register_1_to_1(sigima_image.prewitt_h, _("Prewitt filter (horizontal)"))
        self.register_1_to_1(sigima_image.prewitt_v, _("Prewitt filter (vertical)"))
        self.register_1_to_1(sigima_image.roberts, _("Roberts filter"))
        self.register_1_to_1(sigima_image.scharr, _("Scharr filter"))
        self.register_1_to_1(sigima_image.scharr_h, _("Scharr filter (horizontal)"))
        self.register_1_to_1(sigima_image.scharr_v, _("Scharr filter (vertical)"))
        self.register_1_to_1(sigima_image.sobel, _("Sobel filter"))
        self.register_1_to_1(sigima_image.sobel_h, _("Sobel filter (horizontal)"))
        self.register_1_to_1(sigima_image.sobel_v, _("Sobel filter (vertical)"))

        # Other processing
        self.register_1_to_n(sigima_image.extract_roi, "ROI", icon_name="roi.svg")
        self.register_1_to_1(
            sigima_image.resize,
            _("Resize"),
            sigima_image.ResizeParam,
            icon_name="resize.svg",
        )
        self.register_1_to_1(
            sigima_image.binning,
            _("Pixel binning"),
            sigima_image.BinningParam,
            icon_name="binning.svg",
        )
        self.register_1_to_1(
            sigima_image.resampling,
            _("Resampling"),
            sigima_image.Resampling2DParam,
            icon_name="resampling2d.svg",
        )

    def register_analysis(self) -> None:
        """Register analysis functions."""
        self.register_1_to_0(sigima_image.stats, _("Statistics"), icon_name="stats.svg")
        self.register_1_to_1(
            sigima_image.horizontal_projection,
            _("Horizontal projection"),
            # icon_name="horizontal_projection.svg",
            comment=_(
                "Compute the sum of pixel intensities along each column "
                "(projection on the x-axis)"
            ),
        )
        self.register_1_to_1(
            sigima_image.vertical_projection,
            # icon_name="vertical_projection.svg",
            _("Vertical projection"),
            comment=_(
                "Compute the sum of pixel intensities along each row "
                "(projection on the y-axis)"
            ),
        )

        self.register_1_to_1(
            sigima_image.histogram,
            _("Histogram"),
            paramclass=sigima_base.HistogramParam,
            icon_name="histogram.svg",
        )
        self.register_1_to_0(
            sigima_image.centroid,
            _("Centroid"),
            comment=_("Compute image centroid"),
        )
        self.register_1_to_0(
            sigima_image.enclosing_circle,
            _("Minimum enclosing circle center"),
            comment=_("Compute smallest enclosing circle center"),
        )
        self.register_1_to_0(
            sigima_image.contour_shape,
            _("Contour detection"),
            sigima_image.ContourShapeParam,
            comment=_("Compute contour shape fit"),
        )
        self.register_1_to_0(
            sigima_image.peak_detection,
            _("Peak detection"),
            sigima_image.Peak2DDetectionParam,
            comment=_("Detect peaks in the image"),
        )
        self.register_1_to_0(
            sigima_image.hough_circle_peaks,
            _("Circle Hough transform"),
            sigima_image.HoughCircleParam,
            comment=_("Detect circular shapes using circle Hough transform"),
        )
        # Blob detection
        self.register_1_to_0(
            sigima_image.blob_dog,
            _("Blob detection (DOG)"),
            sigima_image.BlobDOGParam,
            comment=_("Detect blobs using Difference of Gaussian (DOG) method"),
        )
        self.register_1_to_0(
            sigima_image.blob_doh,
            _("Blob detection (DOH)"),
            sigima_image.BlobDOHParam,
            comment=_("Detect blobs using Difference of Gaussian (DOH) method"),
        )
        self.register_1_to_0(
            sigima_image.blob_log,
            _("Blob detection (LOG)"),
            sigima_image.BlobLOGParam,
            comment=_("Detect blobs using Laplacian of Gaussian (LOG) method"),
        )
        self.register_1_to_0(
            sigima_image.blob_opencv,
            _("Blob detection (OpenCV)"),
            sigima_image.BlobOpenCVParam,
            comment=_("Detect blobs using OpenCV SimpleBlobDetector"),
        )

    def create_roi_grid(self) -> None:
        """Create a grid of regions of interest"""
        obj0 = self.panel.objview.get_sel_objects(include_groups=True)[0]
        if any(obj.roi is not None for obj in self.panel.objview.get_sel_objects()):
            if (
                QW.QMessageBox.question(
                    self.panel.parentWidget(),
                    _("Warning"),
                    _(
                        "Creating a ROI grid will overwrite any existing ROI.<br><br>"
                        "Do you want to continue?"
                    ),
                    QW.QMessageBox.Yes | QW.QMessageBox.No,
                    QW.QMessageBox.No,
                )
                == QW.QMessageBox.No
            ):
                return
        editor = ImageGridROIEditor(parent=self.parent(), obj=obj0)
        if exec_dialog(editor):
            for obj in self.panel.objview.get_sel_objects():
                obj.roi = editor.get_roi()
            self.SIG_ADD_SHAPE.emit(get_uuid(obj0))
            self.panel.refresh_plot(
                "selected",
                update_items=True,
                only_visible=False,
                only_existing=True,
            )
            # Now, we ask the user if we shall extract the freshly defined ROI:
            if (
                QW.QMessageBox.question(
                    self.panel.parentWidget(),
                    _("Extract ROI"),
                    _("Do you want to extract images from the defined ROI?"),
                    QW.QMessageBox.Yes | QW.QMessageBox.No,
                    QW.QMessageBox.No,
                )
                == QW.QMessageBox.Yes
            ):
                self.compute_roi_extraction(editor.get_roi())

    @qt_try_except()
    def compute_resize(self, param: sigima.params.ResizeParam | None = None) -> None:
        """Resize image with :py:func:`sigima.proc.image.resize`"""
        obj0 = self.panel.objview.get_sel_objects(include_groups=True)[0]
        for obj in self.panel.objview.get_sel_objects():
            if obj.data.shape != obj0.data.shape:
                QW.QMessageBox.warning(
                    self.panel.parentWidget(),
                    APP_NAME,
                    _("Warning:")
                    + "\n"
                    + _("Selected images do not have the same size"),
                )
        edit, param = self.init_param(param, sigima_image.ResizeParam, _("Resize"))
        if edit:
            original_size = obj0.data.shape
            dlg = ResizeDialog(
                self.plotwidget,
                new_size=original_size,
                old_size=original_size,
                text=_("Destination size:"),
            )
            if not exec_dialog(dlg):
                return
            param.zoom = dlg.get_zoom()
        self.run_feature("resize", param, title=_("Resize"), edit=edit)

    @qt_try_except()
    def compute_binning(self, param: sigima.params.BinningParam | None = None) -> None:
        """Binning image with :py:func:`sigima.proc.image.binning`"""
        edit = param is None
        obj0 = self.panel.objview.get_sel_objects(include_groups=True)[0]
        input_dtype_str = str(obj0.data.dtype)
        title = _("Binning")
        edit, param = self.init_param(param, sigima_image.BinningParam, title)
        if edit:
            param.dtype_str = input_dtype_str
        if param.dtype_str is None:
            param.dtype_str = input_dtype_str
        self.run_feature("binning", param, title=title, edit=edit)

    @qt_try_except()
    def compute_line_profile(
        self, param: sigima.params.LineProfileParam | None = None
    ) -> None:
        """Compute profile along a vertical or horizontal line
        with :py:func:`sigima.proc.image.line_profile`"""
        title = _("Profile")
        add_initial_shape = self.has_param_defaults(sigima.params.LineProfileParam)
        edit, param = self.init_param(param, sigima_image.LineProfileParam, title)
        if edit:
            options = self.panel.plothandler.get_plot_options()
            dlg = ProfileExtractionDialog(
                "line", param, options, self.panel.parent(), add_initial_shape
            )
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg.set_obj(obj)
            if not exec_dialog(dlg):
                return
        self.run_feature("line_profile", param, title=title, edit=False)

    @qt_try_except()
    def compute_segment_profile(
        self, param: sigima.params.SegmentProfileParam | None = None
    ):
        """Compute profile along a segment
        with :py:func:`sigima.proc.image.segment_profile`"""
        title = _("Profile")
        add_initial_shape = self.has_param_defaults(sigima.params.SegmentProfileParam)
        edit, param = self.init_param(param, sigima_image.SegmentProfileParam, title)
        if edit:
            options = self.panel.plothandler.get_plot_options()
            dlg = ProfileExtractionDialog(
                "segment", param, options, self.panel.parent(), add_initial_shape
            )
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg.set_obj(obj)
            if not exec_dialog(dlg):
                return
        self.run_feature("segment_profile", param, title=title, edit=False)

    @qt_try_except()
    def compute_average_profile(
        self, param: sigima.params.AverageProfileParam | None = None
    ) -> None:
        """Compute average profile
        with :py:func:`sigima.proc.image.average_profile`"""
        title = _("Average profile")
        add_initial_shape = self.has_param_defaults(sigima.params.AverageProfileParam)
        edit, param = self.init_param(param, sigima_image.AverageProfileParam, title)
        if edit:
            options = self.panel.plothandler.get_plot_options()
            dlg = ProfileExtractionDialog(
                "rectangle", param, options, self.panel.parent(), add_initial_shape
            )
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            dlg.set_obj(obj)
            if not exec_dialog(dlg):
                return
        self.run_feature("average_profile", param, title=title, edit=False)

    @qt_try_except()
    def compute_radial_profile(
        self, param: sigima.params.RadialProfileParam | None = None
    ) -> None:
        """Compute radial profile
        with :py:func:`sigima.proc.image.radial_profile`"""
        title = _("Radial profile")
        edit, param = self.init_param(param, sigima_image.RadialProfileParam, title)
        if edit:
            obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
            param.update_from_obj(obj)
        self.run_feature("radial_profile", param, title=title, edit=edit)

    @qt_try_except()
    def distribute_on_grid(self, param: sigima.params.GridParam | None = None) -> None:
        """Distribute images on a grid"""
        title = _("Distribute on grid")
        edit, param = self.init_param(param, sigima_image.GridParam, title)
        if edit and not param.edit(parent=self.panel.parent()):
            return
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        g_row, g_col, x0, y0, x0_0, y0_0 = 0, 0, 0.0, 0.0, 0.0, 0.0
        dx0, dy0 = 0.0, 0.0
        with create_progress_bar(self.panel, title, max_=len(objs)) as progress:
            for i_row, obj in enumerate(objs):
                progress.setValue(i_row + 1)
                QW.QApplication.processEvents()
                if progress.wasCanceled():
                    break
                if i_row == 0:
                    x0_0, y0_0 = x0, y0 = obj.x0, obj.y0
                else:
                    dx0, dy0 = x0 - obj.x0, y0 - obj.y0
                    obj.x0 += dx0
                    obj.y0 += dy0
                    apply_geometry_transform(obj, "translate", dx=dx0, dy=dy0)
                    transformer.transform_roi(obj, "translate", dx=dx0, dy=dy0)
                if param.direction == "row":
                    # Distributing images over rows
                    sign = np.sign(param.rows)
                    g_row = (g_row + sign) % param.rows
                    y0 += (obj.height + param.rowspac) * sign
                    if g_row == 0:
                        g_col += 1
                        x0 += obj.width + param.colspac
                        y0 = y0_0
                else:
                    # Distributing images over columns
                    sign = np.sign(param.cols)
                    g_col = (g_col + sign) % param.cols
                    x0 += (obj.width + param.colspac) * sign
                    if g_col == 0:
                        g_row += 1
                        x0 = x0_0
                        y0 += obj.height + param.rowspac
        self.panel.refresh_plot("selected", True, False)

    @qt_try_except()
    def reset_positions(self) -> None:
        """Reset image positions"""
        x0_0, y0_0 = 0.0, 0.0
        dx0, dy0 = 0.0, 0.0
        objs = self.panel.objview.get_sel_objects(include_groups=True)
        for i_row, obj in enumerate(objs):
            if i_row == 0:
                x0_0, y0_0 = obj.x0, obj.y0
            else:
                dx0, dy0 = x0_0 - obj.x0, y0_0 - obj.y0
                obj.x0 += dx0
                obj.y0 += dy0
                apply_geometry_transform(obj, "translate", dx=dx0, dy=dy0)
                transformer.transform_roi(obj, "translate", dx=dx0, dy=dy0)
        self.panel.refresh_plot("selected", True, False)

    # ------Image Processing
    @qt_try_except()
    def compute_offset_correction(self, param: ROI2DParam | None = None) -> None:
        """Compute offset correction
        with :py:func:`sigima.proc.image.offset_correction`"""
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        if param is None:
            dlg = imagebackground.ImageBackgroundDialog(obj, parent=self.panel.parent())
            if exec_dialog(dlg):
                x0, y0, x1, y1 = dlg.get_rect_coords()
                param = ROI2DParam.create(
                    geometry="rectangle",
                    x0=int(x0),
                    y0=int(y0),
                    dx=int(x1 - x0),
                    dy=int(y1 - y0),
                )
            else:
                return
        self.run_feature("offset_correction", param)

    @qt_try_except()
    def compute_erase(self, roi: ImageROI | None = None) -> None:
        """Erase area in the image as defined by a region of interest

        Args:
            roi: Region of interest to erase
        """
        if roi is None or roi.is_empty():
            roi = self.edit_roi_graphically(mode="define")
        if roi is None or roi.is_empty():
            return
        obj = self.panel.objview.get_sel_objects(include_groups=True)[0]
        params = roi.to_params(obj)

        # TODO: This `compute_1_to_1` call is not ideal, as it passes a list of
        # parameter sets (`params` is a list of `DataSet` objects) instead of a single
        # parameter set as expected by the method. Currently, the method implementation
        # is compatible with this call, and it simply passes the second argument through
        # to the `extract_rois` function. However, this should be rectified in the
        # future to ensure that the method signature and its usage are consistent.
        # The question is: should we pass a list of `DataSet` objects or directly a
        # `ImageROI` object?
        # (same as `_extract_multiple_roi_in_single_object`)
        self.compute_1_to_1(
            sigima_image.erase, params, title=_("Erase area"), edit=False
        )

    @qt_try_except()
    def compute_all_threshold(self) -> None:
        """Compute all threshold algorithms
        using the following functions:

        - :py:func:`sigima.proc.image.threshold.threshold_isodata`
        - :py:func:`sigima.proc.image.threshold.threshold_li`
        - :py:func:`sigima.proc.image.threshold.threshold_mean`
        - :py:func:`sigima.proc.image.threshold.threshold_minimum`
        - :py:func:`sigima.proc.image.threshold.threshold_otsu`
        - :py:func:`sigima.proc.image.threshold.threshold_triangle`
        - :py:func:`sigima.proc.image.threshold.threshold_yen`
        """
        self.compute_multiple_1_to_1(
            [
                sigima_image.threshold_isodata,
                sigima_image.threshold_li,
                sigima_image.threshold_mean,
                sigima_image.threshold_minimum,
                sigima_image.threshold_otsu,
                sigima_image.threshold_triangle,
                sigima_image.threshold_yen,
            ],
            None,
            "Threshold",
            edit=False,
        )

    @qt_try_except()
    def compute_all_denoise(self, params: list | None = None) -> None:
        """Compute all denoising filters
        using the following functions:

        - :py:func:`sigima.proc.image.restoration.denoise_tv`
        - :py:func:`sigima.proc.image.restoration.denoise_bilateral`
        - :py:func:`sigima.proc.image.restoration.denoise_wavelet`
        - :py:func:`sigima.proc.image.restoration.denoise_tophat`
        """
        if params is not None:
            assert len(params) == 4, "Wrong number of parameters (4 expected)"
        funcs = [
            sigima_image.denoise_tv,
            sigima_image.denoise_bilateral,
            sigima_image.denoise_wavelet,
            sigima_image.denoise_tophat,
        ]
        edit = params is None
        if edit:
            params = []
            for paramclass, title in (
                (sigima_image.DenoiseTVParam, _("Total variation denoising")),
                (sigima_image.DenoiseBilateralParam, _("Bilateral filter denoising")),
                (sigima_image.DenoiseWaveletParam, _("Wavelet denoising")),
                (sigima_image.MorphologyParam, _("Denoise / Top-Hat")),
            ):
                param = paramclass(title)
                self.update_param_defaults(param)
                params.append(param)
        self.compute_multiple_1_to_1(funcs, params, "Denoise", edit=edit)

    @qt_try_except()
    def compute_all_morphology(
        self, param: sigima.params.MorphologyParam | None = None
    ) -> None:
        """Compute all morphology filters
        using the following functions:

        - :py:func:`sigima.proc.image.morphology.white_tophat`
        - :py:func:`sigima.proc.image.morphology.black_tophat`
        - :py:func:`sigima.proc.image.morphology.erosion`
        - :py:func:`sigima.proc.image.morphology.dilation`
        - :py:func:`sigima.proc.image.morphology.opening`
        - :py:func:`sigima.proc.image.morphology.closing`
        """
        if param is None:
            param = sigima_image.MorphologyParam()
            if not param.edit(parent=self.panel.parent()):
                return
        funcs = [
            sigima_image.white_tophat,
            sigima_image.black_tophat,
            sigima_image.erosion,
            sigima_image.dilation,
            sigima_image.opening,
            sigima_image.closing,
        ]
        self.compute_multiple_1_to_1(funcs, [param] * len(funcs), "Morph", edit=False)

    @qt_try_except()
    def compute_all_edges(self) -> None:
        """Compute all edge detection algorithms.

        This function calls the following edge detection algorithms:

        - :py:func:`sigima.proc.image.edges.canny`
        - :py:func:`sigima.proc.image.edges.farid`
        - :py:func:`sigima.proc.image.edges.farid_h`
        - :py:func:`sigima.proc.image.edges.farid_v`
        - :py:func:`sigima.proc.image.edges.laplace`
        - :py:func:`sigima.proc.image.edges.prewitt`
        - :py:func:`sigima.proc.image.edges.prewitt_h`
        - :py:func:`sigima.proc.image.edges.prewitt_v`
        - :py:func:`sigima.proc.image.edges.roberts`
        - :py:func:`sigima.proc.image.edges.scharr`
        - :py:func:`sigima.proc.image.edges.scharr_h`
        - :py:func:`sigima.proc.image.edges.scharr_v`
        - :py:func:`sigima.proc.image.edges.sobel`
        - :py:func:`sigima.proc.image.edges.sobel_h`
        - :py:func:`sigima.proc.image.edges.sobel_v`
        """
        funcs = [
            sigima_image.canny,
            sigima_image.farid,
            sigima_image.farid_h,
            sigima_image.farid_v,
            sigima_image.laplace,
            sigima_image.prewitt,
            sigima_image.prewitt_h,
            sigima_image.prewitt_v,
            sigima_image.roberts,
            sigima_image.scharr,
            sigima_image.scharr_h,
            sigima_image.scharr_v,
            sigima_image.sobel,
            sigima_image.sobel_h,
            sigima_image.sobel_v,
        ]
        self.compute_multiple_1_to_1(funcs, None, "Edges")

    @qt_try_except()
    def _extract_multiple_roi_in_single_object(self, params: list[ROI2DParam]) -> None:
        """Extract multiple Regions Of Interest (ROIs) from data in a single object"""
        # TODO: This `compute_1_to_1` call is not ideal, as it passes a list of
        # parameter sets (`params` is a list of `DataSet` objects) instead of a single
        # parameter set as expected by the method. Currently, the method implementation
        # is compatible with this call, and it simply passes the second argument through
        # to the `extract_rois` function. However, this should be rectified in the
        # future to ensure that the method signature and its usage are consistent.
        # The question is: should we pass a list of `DataSet` objects or directly a
        # `ImageROI` object?
        # (same as `compute_erase`)
        self.compute_1_to_1(sigima_image.extract_rois, params, title=_("Extract ROI"))

    # ------Image Analysis
    @qt_try_except()
    def compute_peak_detection(
        self, param: sigima.params.Peak2DDetectionParam | None = None
    ) -> dict[str, GeometryResult]:
        """Compute 2D peak detection
        with :py:func:`sigima.proc.image.peak_detection`"""
        edit, param = self.init_param(
            param, sigima_image.Peak2DDetectionParam, _("Peak detection")
        )
        if edit:
            data = self.panel.objview.get_sel_objects(include_groups=True)[0].data
            param.size = max(min(data.shape) // 40, 50)

        results = self.run_feature("peak_detection", param, edit=edit)
        return results
