Menus
=====

This section describes the image related features of DataLab, by presenting
the different menus and their associated functions.

"File" menu
-----------

.. figure:: /images/shots/i_file.png

    Screenshot of the "File" menu.

The "File" menu allows you to create, open, save and close images. It also
allows you to import and export data from/to HDF5 files, and to edit the
settings of the current session.

New image
^^^^^^^^^

Create a new image from various models
(supported datatypes: uint8, uint16, int16, float32, float64):

.. list-table::
    :header-rows: 1
    :widths: 20, 80

    * - Model
      - Equation
    * - Zeros
      - :math:`z[i] = 0`
    * - Empty
      - Data is directly taken from memory as it is
    * - Random
      - :math:`z[i] \in [0, z_{max})` where :math:`z_{max}` is the datatype maximum value
    * - 2D Gaussian
      - :math:`z = A.exp(-\dfrac{(\sqrt{(x-x0)^2+(y-y0)^2}-\mu)^2}{2\sigma^2})`

.. _open_image:

Open image
^^^^^^^^^^

Create a new image from the following supported filetypes:

.. list-table::
    :header-rows: 1

    * - File type
      - Extensions
    * - PNG files
      - .png
    * - TIFF files
      - .tif, .tiff
    * - 8-bit images
      - .jpg, .gif
    * - NumPy arrays
      - .npy
    * - Text files
      - .txt, .csv, .asc
    * - Andor SIF files
      - .sif
    * - SPIRICON files
      - .scor-data
    * - FXD files
      - .fxd
    * - Bitmap images
      - .bmp

Save image
^^^^^^^^^^

Save current image (see "Open image" supported filetypes).

Open HDF5 file
^^^^^^^^^^^^^^

Import data from a HDF5 file.

Save to HDF5 file
^^^^^^^^^^^^^^^^^

Export the whole DataLab session (all signals and images) into a HDF5 file.

Browse HDF5 file
^^^^^^^^^^^^^^^^

Open the :ref:`h5browser` in a new window to browse and import data from HDF5 file.

Settings
^^^^^^^^

Open the the "Settings" dialog box.

.. image:: /images/settings.png

"Edit" menu
-----------

.. figure:: /images/shots/i_edit.png

    Screenshot of the "Edit" menu.

The "Edit" menu allows you to edit the current image or group of images, by
adding, removing, renaming, moving up or down, or duplicating images. It also
manipulates metadata, or handles image titles.

New group
^^^^^^^^^

Create a new group of images. Images may be moved from one group to another
by drag and drop.

Rename group
^^^^^^^^^^^^

Rename currently selected group.

Move up
^^^^^^^

Move current selection up in the list (groups or images may be selected). If
multiple objects are selected, they are moved together. If a selected image
is already at the top of its group, it is moved to the bottom of the previous
group.

Move down
^^^^^^^^^

Move current selection down in the list (groups or images may be selected). If
multiple objects are selected, they are moved together. If a selected image
is already at the bottom of its group, it is moved to the top of the next
group.

Duplicate
^^^^^^^^^

Create a new image which is identical to the currently selected object.

Remove
^^^^^^

Remove currently selected image.

Delete all
^^^^^^^^^^

Delete all images.

Copy metadata
^^^^^^^^^^^^^

Copy metadata from currently selected image into clipboard.

Paste metadata
^^^^^^^^^^^^^^

Paste metadata from clipboard into selected image.

Import metadata into image
^^^^^^^^^^^^^^^^^^^^^^^^^^

Import metadata from a JSON text file.

Export metadata from image
^^^^^^^^^^^^^^^^^^^^^^^^^^

Export metadata to a JSON text file.

Delete object metadata
^^^^^^^^^^^^^^^^^^^^^^

Delete metadata from currently selected image.
Metadata contains additionnal information such as Region of Interest
or results of computations

Add object title to plot
^^^^^^^^^^^^^^^^^^^^^^^^

Add currently selected image title to the associated plot.

Copy titles to clipboard
^^^^^^^^^^^^^^^^^^^^^^^^

Copy all image titles to clipboard as a multiline text.
This text may be used for reproducing a processing chain, for example.


"Operation" menu
----------------

.. figure:: /images/shots/i_operation.png

    Screenshot of the "Operation" menu.

The "Operation" menu allows you to perform various operations on the current
image or group of images. It also allows you to extract profiles, distribute
images on a grid, or resize images.

Sum
^^^

Create a new image which is the sum of all selected images:

.. math::
    z_{M} = \sum_{k=0}^{M-1}{z_{k}}

Average
^^^^^^^

Create a new image which is the average of all selected images:

.. math::
    z_{M} = \dfrac{1}{M}\sum_{k=0}^{M-1}{z_{k}}

Difference
^^^^^^^^^^

Create a new image which is the difference of the **two** selected images:

.. math::
    z_{2} = z_{1} - z_{0}

Quadratic difference
^^^^^^^^^^^^^^^^^^^^

Create a new image which is the quadratic difference of the **two**
selected images:

.. math::
    z_{2} = \dfrac{z_{1} - z_{0}}{\sqrt{2}}

Product
^^^^^^^

Create a new image which is the product of all selected images:

.. math::
    z_{M} = \prod_{k=0}^{M-1}{z_{k}}

Division
^^^^^^^^

Create a new image which is the division of the **two** selected images:

.. math::
    z_{2} = \dfrac{z_{1}}{z_{0}}

Absolute value
^^^^^^^^^^^^^^

Create a new image which is the absolute value of each selected image:

.. math::
    z_{k} = |z_{k-1}|

Real part
^^^^^^^^^

Create a new image which is the real part of each selected image:

.. math::
    z_{k} = \Re(z_{k-1})

Imaginary part
^^^^^^^^^^^^^^

Create a new image which is the imaginary part of each selected image:

.. math::
    z_{k} = \Im(z_{k-1})

Convert data type
^^^^^^^^^^^^^^^^^

Create a new image which is the result of converting data type of each
selected image.

.. note::

    Data type conversion relies on :py:func:`numpy.ndarray.astype` function with
    the default parameters (`casting='unsafe'`).

Log10(z)
^^^^^^^^

Create a new image which is the base 10 logarithm of each selected image:

.. math::
    z_{k} = \log_{10}(z_{k-1})

Log10(z+n)
^^^^^^^^^^

Create a new image which is the Log10(z+n) of each selected image
(avoid Log10(0) on image background):

.. math::
    z_{k} = \log_{10}(z_{k-1}+n)

Flat-field correction
^^^^^^^^^^^^^^^^^^^^^

Create a new image which is flat-field correction
of the **two** selected images:

.. math::
    z_{1} =
    \begin{cases}
        \dfrac{z_{0}}{z_{f}}.\overline{z_{f}} & \text{if } z_{0} > z_{threshold} \\
        z_{0} & \text{otherwise}
    \end{cases}`

where :math:`z_{0}` is the raw image,
:math:`z_{f}` is the flat field image,
:math:`z_{threshold}` is an adjustable threshold
and :math:`\overline{z_{f}}` is the flat field image average value:

.. math::
    \overline{z_{f}}=
    \dfrac{1}{N_{row}.N_{col}}.\sum_{i=0}^{N_{row}}\sum_{j=0}^{N_{col}}{z_{f}(i,j)}

.. note::

    Raw image and flat field image are supposedly already
    corrected by performing a dark frame subtraction.

Rotation
^^^^^^^^

Create a new image which is the result of rotating (90°, 270° or
arbitrary angle) or flipping (horizontally or vertically) data.

Intensity profiles
^^^^^^^^^^^^^^^^^^

Line profile
    Extract an horizontal or vertical profile from each selected image, and create
    new signals from these profiles.

    .. figure:: /images/shots/i_profile.png

        Line profile dialog. Parameters may also be set manually
        ("Edit profile parameters" button).

Average profile
    Extract an horizontal or vertical profile averaged over a rectangular area, from
    each selected image, and create new signals from these profiles.

    .. figure:: /images/shots/i_profile_average.png

        Average profile dialog: the area is defined by a rectangle shape.
        Parameters may also be set manually ("Edit profile parameters" button).

Radial profile extraction
    Extract a radial profile from each selected image, and create new signals from
    these profiles.

    The following parameters are available:

    .. list-table::
        :header-rows: 1
        :widths: 25, 75

        * - Parameter
          - Description
        * - Center
          - Center around which the radial profile is computed: centroid, image center, or user-defined
        * - X
          - X coordinate of the center (if user-defined), in pixels
        * - Y
          - Y coordinate of the center (if user-defined), in pixels

Distribute on a grid
^^^^^^^^^^^^^^^^^^^^

Distribute selected images on a regular grid.

Reset image positions
^^^^^^^^^^^^^^^^^^^^^

Reset selected image positions to first image (x0, y0) coordinates.

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

Swap X/Y axes
^^^^^^^^^^^^^

Create a new image which is the result of swapping X/Y data.

"Processing" menu
-----------------

.. figure:: /images/shots/i_processing.png

    Screenshot of the "Processing" menu.

The "Processing" menu allows you to perform various processing on the current
image or group of images: it allows you to apply filters, to perform exposure
correction, to perform denoising, to perform morphological operations, and so on.

Linear calibration
^^^^^^^^^^^^^^^^^^

Create a new image which is a linear calibration
of each selected image with respect to Z axis:

.. list-table::
    :header-rows: 1
    :widths: 40, 60

    * - Parameter
      - Linear calibration
    * - Z-axis
      - :math:`z_{1} = a.z_{0} + b`

Thresholding
^^^^^^^^^^^^

Apply the thresholding to each selected image.

Clipping
^^^^^^^^

Apply the clipping to each selected image.

Moving average
^^^^^^^^^^^^^^

Compute moving average of each selected image
(implementation based on `scipy.ndimage.uniform_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html>`_).

Moving median
^^^^^^^^^^^^^

Compute moving median of each selected image
(implementation based on `scipy.signal.medfilt <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html>`_).

Wiener filter
^^^^^^^^^^^^^

Compute Wiener filter of each selected image
(implementation based on `scipy.signal.wiener <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html>`_).

FFT
^^^

Create a new image which is the Fast Fourier Transform (FFT)
of each selected image.

Inverse FFT
^^^^^^^^^^^

Create a new image which is the inverse FFT of each selected image.

Butterworth filter
^^^^^^^^^^^^^^^^^^

Perform Butterworth filter on an image
(implementation based on `skimage.filters.butterworth <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.butterworth>`_)

Exposure
^^^^^^^^

Gamma correction
    Apply gamma correction to each selected image
    (implementation based on `skimage.exposure.adjust_gamma <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma>`_)

Logarithmic correction
    Apply logarithmic correction to each selected image
    (implementation based on `skimage.exposure.adjust_log <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_log>`_)

Sigmoid correction
    Apply sigmoid correction to each selected image
    (implementation based on `skimage.exposure.adjust_sigmoid <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_sigmoid>`_)

Histogram equalization
    Equalize image histogram levels
    (implementation based on `skimage.exposure.equalize_hist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist>`_)

Adaptive histogram equalization
    Equalize image histogram levels using Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm
    (implementation based on `skimage.exposure.equalize_adapthist <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist>`_)

Intensity rescaling
    Stretch or shrink image intensity levels
    (implementation based on `skimage.exposure.rescale_intensity <https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity>`_)

Restoration
^^^^^^^^^^^

Total variation denoising
    Denoise image using Total Variation algorithm
    (implementation based on `skimage.restoration.denoise_tv_chambolle <https://scikit-image.org/docs/stable/api/skimage.restoration.html#denoise-tv-chambolle>`_)

Bilateral filter denoising
    Denoise image using bilateral filter
    (implementation based on `skimage.restoration.denoise_bilateral <https://scikit-image.org/docs/stable/api/skimage.restoration.html#denoise-bilateral>`_)

Wavelet denoising
    Perform wavelet denoising on image
    (implementation based on `skimage.restoration.denoise_wavelet <https://scikit-image.org/docs/stable/api/skimage.restoration.html#denoise-wavelet>`_)

White Top-Hat denoising
    Denoise image by subtracting its white top hat transform
    (using a disk footprint)

All denoising methods
    Perform all denoising methods on image. Combined with the
    "distribute on a grid" option, this allows to compare the
    different denoising methods on the same image.

Morphology
^^^^^^^^^^

White Top-Hat (disk)
    Perform white top hat transform of an image, using a disk footprint
    (implementation based on `skimage.morphology.white_tophat <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.white_tophat>`_)

Black Top-Hat (disk)
    Perform black top hat transform of an image, using a disk footprint
    (implementation based on `skimage.morphology.black_tophat <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.black_tophat>`_)

Erosion (disk)
    Perform morphological erosion on an image, using a disk footprint
    (implementation based on `skimage.morphology.erosion <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion>`_)

Dilation (disk)
    Perform morphological dilation on an image, using a disk footprint
    (implementation based on `skimage.morphology.dilation <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation>`_)

Opening (disk)
    Perform morphological opening on an image, using a disk footprint
    (implementation based on `skimage.morphology.opening <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.opening>`_)

Closing (disk)
    Perform morphological closing on an image, using a disk footprint
    (implementation based on `skimage.morphology.closing <https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.closing>`_)

All morphological operations
    Perform all morphological operations on an image, using a disk footprint.
    Combined with the "distribute on a grid" option, this allows to compare
    the different morphological operations on the same image.

Edges
^^^^^

Roberts filter
    Perform edge filtering on an image, using the Roberts algorithm
    (implementation based on `skimage.filters.roberts <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts>`_)

Prewitt filter
    Perform edge filtering on an image, using the Prewitt algorithm
    (implementation based on `skimage.filters.prewitt <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt>`_)

Prewitt filter (horizontal)
    Find the horizontal edges of an image, using the Prewitt algorithm
    (implementation based on `skimage.filters.prewitt_h <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_h>`_)

Prewitt filter (vertical)
    Find the vertical edges of an image, using the Prewitt algorithm
    (implementation based on `skimage.filters.prewitt_v <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_v>`_)

Sobel filter
    Perform edge filtering on an image, using the Sobel algorithm
    (implementation based on `skimage.filters.sobel <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel>`_)

Sobel filter (horizontal)
    Find the horizontal edges of an image, using the Sobel algorithm
    (implementation based on `skimage.filters.sobel_h <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_h>`_)

Sobel filter (vertical)
    Find the vertical edges of an image, using the Sobel algorithm
    (implementation based on `skimage.filters.sobel_v <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_v>`_)

Scharr filter
    Perform edge filtering on an image, using the Scharr algorithm
    (implementation based on `skimage.filters.scharr <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr>`_)

Scharr filter (horizontal)
    Find the horizontal edges of an image, using the Scharr algorithm
    (implementation based on `skimage.filters.scharr_h <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_h>`_)

Scharr filter (vertical)
    Find the vertical edges of an image, using the Scharr algorithm
    (implementation based on `skimage.filters.scharr_v <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_v>`_)

Farid filter
    Perform edge filtering on an image, using the Farid algorithm
    (implementation based on `skimage.filters.farid <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid>`_)

Farid filter (horizontal)
    Find the horizontal edges of an image, using the Farid algorithm
    (implementation based on `skimage.filters.farid_h <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_h>`_)

Farid filter (vertical)
    Find the vertical edges of an image, using the Farid algorithm
    (implementation based on `skimage.filters.farid_v <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_v>`_)

Laplace filter
    Perform edge filtering on an image, using the Laplace algorithm
    (implementation based on `skimage.filters.laplace <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.laplace>`_)

All edges filters
    Perform all edge filtering algorithms (see above) on an image.
    Combined with the "distribute on a grid" option, this allows to compare
    the different edge filters on the same image.

Canny filter
    Perform edge filtering on an image, using the Canny algorithm
    (implementation based on `skimage.feature.canny <https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny>`_)

"Computing" menu
----------------

.. figure:: /images/shots/i_computing.png

    Screenshot of the "Computing" menu.

The "Computing" menu allows you to perform various computations on the current
image or group of images. It also allows you to compute statistics, to compute
the centroid, to detect peaks, to detect contours, and so on.

.. note::

    In DataLab vocabulary, a "computing" is a feature that computes a scalar
    result from an image. This result is stored as metadata, and thus attached
    to image. This is different from a "processing" which creates a new image
    from an existing one.

Edit regions of interest
^^^^^^^^^^^^^^^^^^^^^^^^

Open a dialog box to setup multiple Region Of Interests (ROI).
ROI are stored as metadata, and thus attached to image.

ROI definition dialog is exactly the same as ROI extraction (see above).

.. figure:: /images/shots/i_roi_image.png

    An image with ROI.

Remove regions of interest
^^^^^^^^^^^^^^^^^^^^^^^^^^

Remove all defined ROI for selected object(s).

Statistics
^^^^^^^^^^

Compute statistics on selected image and show a summary table.

.. figure:: /images/shots/i_stats.png

    Example of statistical summary table: each row is associated to an ROI
    (the first row gives the statistics for the whole data).

Centroid
^^^^^^^^

Compute image centroid using a Fourier transform method
(as discussed by `Weisshaar et al. <http://www.mnd-umwelttechnik.fh-wiesbaden.de/pig/weisshaar_u5.pdf>`_).
This method is quite insensitive to background noise.

Minimum enclosing circle center
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute the circle contour enclosing image values above
a threshold level defined as the half-maximum value.

2D peak detection
^^^^^^^^^^^^^^^^^

Automatically find peaks on image using a minimum-maximum filter algorithm.

.. figure:: /images/shots/i_peak2d_test.png

    Example of 2D peak detection.

.. seealso::
    See :ref:`ref-to-2d-peak-detection` for more details on algorithm and associated parameters.

Contour detection
^^^^^^^^^^^^^^^^^

Automatically extract contours and fit them using a circle or an ellipse,
or directly represent them as a polygon.

.. figure:: /images/shots/i_contour_test.png

    Example of contour detection.

.. seealso::
    See :ref:`ref-to-contour-detection` for more details on algorithm and associated parameters.

.. note:: Computed scalar results are systematically stored as metadata.
    Metadata is attached to image and serialized with it when exporting
    current session in a HDF5 file.

Circle Hough transform
^^^^^^^^^^^^^^^^^^^^^^

Detect circular shapes using circle Hough transform
(implementation based on `skimage.transform.hough_circle_peaks <https://scikit-image.org/docs/stable/api/skimage.transform.html?highlight=hough#skimage.transform.hough_circle_peaks>`_).

Blob detection
^^^^^^^^^^^^^^

Blob detection (DOG)
    Detect blobs using Difference of Gaussian (DOG) method
    (implementation based on `skimage.feature.blob_dog <https://scikit-image.org/docs/stable/api/skimage.feature.html#blob-dog>`_).

Blob detection (DOH)
    Detect blobs using Determinant of Hessian (DOH) method
    (implementation based on `skimage.feature.blob_doh <https://scikit-image.org/docs/stable/api/skimage.feature.html#blob-doh>`_).

Blob detection (LOG)
    Detect blobs using Laplacian of Gaussian (LOG) method
    (implementation based on `skimage.feature.blob_log <https://scikit-image.org/docs/stable/api/skimage.feature.html#blob-log>`_).

Blob detection (OpenCV)
    Detect blobs using OpenCV implementation of `SimpleBlobDetector <https://docs.opencv.org/3.4/d0/d7a/classcv_1_1SimpleBlobDetector.html>`_.

Show results
^^^^^^^^^^^^

Show the results of all computations performed on the selected images. This shows the
same table as the one shown after having performed a computation.

Plot results
^^^^^^^^^^^^

Plot the results of computations performed on the selected images, with user-defined
X and Y axes (e.g. plot the contour circle radius as a function of the image number).

"View" menu
-----------

.. figure:: /images/shots/i_view.png

    Screenshot of the "View" menu.

The "View" menu allows you to visualize the current image or group of images.
It also allows you to show/hide titles, to show/hide the contrast panel, to
refresh the visualization, and so on.

View in a new window
^^^^^^^^^^^^^^^^^^^^

Open a new window to visualize and the selected images.

In the separate window, you may visualize your data more comfortably
(e.g., by maximizing the window) and you may also annotate the data.

.. seealso::
    See :ref:`ref-to-image-annotations` for more details on annotations.

Show graphical object titles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Show/hide titles of computing results or annotations.

Auto-refresh
^^^^^^^^^^^^

Automatically refresh the visualization when the data changes.
When enabled (default), the plot view is automatically refreshed when the
data changes. When disabled, the plot view is not refreshed until you
manually refresh it by clicking the "Refresh manually" button in the
toolbar. Even though the refresh algorithm is optimized, it may still
take some time to refresh the plot view when the data changes, especially
when the data set is large. Therefore, you may want to disable the
auto-refresh feature when you are working with large data sets,
and enable it again when you are done. This will avoid unnecessary
refreshes.

Refresh manually
^^^^^^^^^^^^^^^^

Refresh the visualization manually. This triggers a refresh of the plot
view, even if the auto-refresh feature is disabled.

Show contrast panel
^^^^^^^^^^^^^^^^^^^

Show/hide contrast adjustment panel.

Other menu entries
^^^^^^^^^^^^^^^^^^

Show/hide panels or toolbars.

"?" menu
--------

.. figure:: /images/shots/s_help.png

    Screenshot of the "?" menu.

The "?" menu allows you to access the online documentation, to show log files,
to show information regarding your DataLab installation, and to show the
"About DataLab" dialog box.

Online or Local documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open the online or local documentation:

.. image:: /images/shots/doc_online.png

Show log files
^^^^^^^^^^^^^^

Open DataLab log viewer

.. seealso::
    See :ref:`ref-to-logviewer` for more details on log viewer.

About DataLab installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Show information regarding your DataLab installation (this
is typically needed for submitting a bug report).

.. seealso::
    See :ref:`ref-to-instviewer` for more details on this dialog box.

About
^^^^^

Open the "About DataLab" dialog box:

.. image:: /images/shots/about.png
