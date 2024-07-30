.. _ima-menu-processing:

Processing Images
=================

This section describes the image processing features available in DataLab.

.. seealso::

    :ref:`ima-menu-operations` for more information on operations that can be performed
    on images, or :ref:`ima-menu-analysis` for information on analysis features on
    images.

.. figure:: /images/shots/i_processing.png

    Screenshot of the "Processing" menu.

When the "Image Panel" is selected, the menus and toolbars are updated to
provide image-related actions.

The "Processing" menu allows you to perform various processing on the current
image or group of images: it allows you to apply filters, to perform exposure
correction, to perform denoising, to perform morphological operations, and so on.

Axis transformation
^^^^^^^^^^^^^^^^^^^

Linear calibration
~~~~~~~~~~~~~~~~~~

Create a new image which is a linear calibration
of each selected image with respect to Z axis:

.. list-table::
    :header-rows: 1
    :widths: 40, 60

    * - Parameter
      - Linear calibration
    * - Z-axis
      - :math:`z_{1} = a.z_{0} + b`

Swap X/Y axes
~~~~~~~~~~~~~

Create a new image which is the result of swapping X/Y data.

Level adjustment
^^^^^^^^^^^^^^^^

Normalize
~~~~~~~~~

Create a new image which is the normalized version of each selected image
by maximum, amplitude, sum, energy or RMS:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Normalization
      - Equation
    * - Maximum
      - :math:`z_{1} = \dfrac{z_{0}}{z_{max}}`
    * - Amplitude
      - :math:`z_{1} = \dfrac{z_{0}}{z_{max}-z_{min}}`
    * - Area
      - :math:`z_{1} = \dfrac{z_{0}}{\sum_{i=0}^{N-1}{z_{i}}}`
    * - Energy
      - :math:`z_{1}= \dfrac{z_{0}}{\sqrt{\sum_{n=0}^{N}|z_{0}[n]|^2}}`
    * - RMS
      - :math:`z_{1}= \dfrac{z_{0}}{\sqrt{\dfrac{1}{N}\sum_{n=0}^{N}|z_{0}[n]|^2}}`

Clipping
~~~~~~~~

Apply the clipping to each selected image.

Offset correction
~~~~~~~~~~~~~~~~~

Create a new image which is the result of offset correction on each selected image.
This operation is performed by subtracting the image background value which is estimated
by the mean value of a user-defined rectangular area.

Noise reduction
^^^^^^^^^^^^^^^

Create a new image which is the result of noise reduction on each selected image.

The following filters are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Filter
      - Formula/implementation
    * - Gaussian filter
      - `scipy.ndimage.gaussian_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html>`_
    * - Moving average
      - `scipy.ndimage.uniform_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html>`_
    * - Moving median
      - `scipy.ndimage.median_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html>`_
    * - Wiener filter
      - `scipy.signal.wiener <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html>`_

Fourier analysis
^^^^^^^^^^^^^^^^

Create a new image which is the result of a Fourier analysis on each selected image.

The following functions are available:

.. list-table::
    :header-rows: 1
    :widths: 20, 30, 50

    * - Function
      - Description
      - Formula/implementation
    * - FFT
      - Fast Fourier Transform
      - `numpy.fft.fft2 <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html>`_
    * - Inverse FFT
      - Inverse Fast Fourier Transform
      - `numpy.fft.ifft2 <https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html>`_
    * - Magnitude spectrum
      - Optionnal: use logarithmic scale (dB)
      - :math:`z_{1} = |FFT(z_{0})|` or :math:`z_{1} = 20 \log_{10}(|FFT(z_{0})|)` (dB)
    * - Phase spectrum
      -
      - :math:`z_{1} = \angle(FFT(z_{0}))`
    * - Power spectral density
      - Optionnal: use logarithmic scale (dB)
      - :math:`z_{1} = |FFT(z_{0})|^2` or :math:`z_{1} = 10 \log_{10}(|FFT(z_{0})|^2)` (dB)

.. note::

    FFT and inverse FFT are performed using frequency shifting if the option is enabled
    in DataLab settings (see :ref:`settings`).

Thresholding
^^^^^^^^^^^^

Create a new image which is the result of thresholding on each selected image,
eventually based on user-defined parameters ("Parametric thresholding").

The following parameters are available when selecting "Parametric thresholding":

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Parameter
      - Description
    * - Threshold method
      - The thresholding method to use (see table below)
    * - Bins
      - Number of bins for histogram calculation
    * - Value
      - Threshold value
    * - Operation
      - Operation to apply (> or <)

The following thresholding methods are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Method
      - Implementation
    * - Manual
      - Manual thresholding (user-defined parameters)
    * - ISODATA
      - `skimage.filters.threshold_isodata <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_isodata>`_
    * - Li
      - `skimage.filters.threshold_li <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_li>`_
    * - Mean
      - `skimage.filters.threshold_mean <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_mean>`_
    * - Minimum
      - `skimage.filters.threshold_minimum <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_minimum>`_
    * - Otsu
      - `skimage.filters.threshold_otsu <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu>`_
    * - Triangle
      - `skimage.filters.threshold_triangle <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_triangle>`_
    * - Yen
      - `skimage.filters.threshold_yen <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_yen>`_

.. note::

    The "All thresholding methods" option allows to perform all thresholding methods
    on the same image. Combined with the "distribute on a grid" option, this allows
    to compare the different thresholding methods on the same image.

Exposure
^^^^^^^^

Create a new image which is the result of exposure correction on each selected image.

The following functions are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 25, 50

    * - Function
      - Implementation
      - Comments
    * - Gamma correction
      - `skimage.exposure.adjust_gamma <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma>`_
      -
    * - Logarithmic correction
      - `skimage.exposure.adjust_log <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_log>`_
      -
    * - Sigmoid correction
      - `skimage.exposure.adjust_sigmoid <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_sigmoid>`_
      -
    * - Histogram equalization
      - `skimage.exposure.equalize_hist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist>`_
      -
    * - Adaptive histogram equalization
      - `skimage.exposure.equalize_adapthist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist>`_
      - Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm
    * - Intensity rescaling
      - `skimage.exposure.rescale_intensity <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity>`_
      - Stretch or shrink image intensity levels

Restoration
^^^^^^^^^^^

Create a new image which is the result of restoration on each selected image.

The following functions are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 25, 50

    * - Function
      - Implementation
      - Comments
    * - Total variation denoising
      - `skimage.restoration.denoise_tv_chambolle <https://scikit-image.org/docs/stable/api/skimage.restoration.html#denoise-tv-chambolle>`_
      -
    * - Bilateral filter denoising
      - `skimage.restoration.denoise_bilateral <https://scikit-image.org/docs/stable/api/skimage.restoration.html#denoise-bilateral>`_
      -
    * - Wavelet denoising
      - `skimage.restoration.denoise_wavelet <https://scikit-image.org/docs/stable/api/skimage.restoration.html#denoise-wavelet>`_
      -
    * - White Top-Hat denoising
      - `skimage.morphology.white_tophat <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.white_tophat>`_
      - Denoise image by subtracting its white top hat transform

.. note::

    The "All denoising methods" option allows to perform all denoising methods
    on the same image. Combined with the "distribute on a grid" option, this allows
    to compare the different denoising methods on the same image.

Morphology
^^^^^^^^^^

Create a new image which is the result of morphological operations on each selected image,
using a disk footprint.

The following functions are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Function
      - Implementation
    * - White Top-Hat (disk)
      - `skimage.morphology.white_tophat <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.white_tophat>`_
    * - Black Top-Hat (disk)
      - `skimage.morphology.black_tophat <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.black_tophat>`_
    * - Erosion (disk)
      - `skimage.morphology.erosion <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion>`_
    * - Dilation (disk)
      - `skimage.morphology.dilation <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation>`_
    * - Opening (disk)
      - `skimage.morphology.opening <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.opening>`_
    * - Closing (disk)
      - `skimage.morphology.closing <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.closing>`_

.. note::

    The "All morphological operations" option allows to perform all morphological operations
    on the same image. Combined with the "distribute on a grid" option, this allows to compare
    the different morphological operations on the same image.

Edges
^^^^^

Create a new image which is the result of edge filtering on each selected image.

The following functions are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Function
      - Implementation
    * - Roberts filter
      - `skimage.filters.roberts <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts>`_
    * - Prewitt filter
      - `skimage.filters.prewitt <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt>`_
    * - Prewitt filter (horizontal)
      - `skimage.filters.prewitt_h <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_h>`_
    * - Prewitt filter (vertical)
      - `skimage.filters.prewitt_v <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_v>`_
    * - Sobel filter
      - `skimage.filters.sobel <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel>`_
    * - Sobel filter (horizontal)
      - `skimage.filters.sobel_h <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_h>`_
    * - Sobel filter (vertical)
      - `skimage.filters.sobel_v <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_v>`_
    * - Scharr filter
      - `skimage.filters.scharr <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr>`_
    * - Scharr filter (horizontal)
      - `skimage.filters.scharr_h <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_h>`_
    * - Scharr filter (vertical)
      - `skimage.filters.scharr_v <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_v>`_
    * - Farid filter
      - `skimage.filters.farid <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters>`_
    * - Farid filter (horizontal)
      - `skimage.filters.farid_h <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_h>`_
    * - Farid filter (vertical)
      - `skimage.filters.farid_v <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_v>`_
    * - Laplace filter
      - `skimage.filters.laplace <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.laplace>`_
    * - Canny filter
      - `skimage.feature.canny <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny>`_

.. note::

    The "All edges filters" option allows to perform all edge filtering algorithms
    on the same image. Combined with the "distribute on a grid" option, this allows
    to compare the different edge filters on the same image.

Butterworth filter
^^^^^^^^^^^^^^^^^^

Perform Butterworth filter on an image
(implementation based on `skimage.filters.butterworth <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.butterworth>`_)

Resize
^^^^^^

Create a new image which is a resized version of each selected image.

Pixel binning
^^^^^^^^^^^^^

Combine clusters of adjacent pixels, throughout the image,
into single pixels. The result can be the sum, average, median, minimum,
or maximum value of the cluster.

ROI extraction
^^^^^^^^^^^^^^

Create a new image from a user-defined Region of Interest.

.. figure:: /images/shots/i_roi_dialog.png

    ROI extraction dialog: the ROI is defined by moving the position
    and adjusting the size of a rectangle shape.
