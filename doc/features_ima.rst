Image processing
================

.. figure:: /images/shots/i_simple_example.png

    CodraFT main window: Image processing view

"File" menu
-----------

.. image:: /images/shots/i_file.png

New image
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

Open image
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
    Save current image (see "Open image" supported filetypes).

Import metadata into image
    Import metadata from a JSON text file.

Export metadata from image
    Export metadata to a JSON text file.

Open HDF5 file
    Import data from a HDF5 file.

Save to HDF5 file
    Export the whole CodraFT session (all signals and images) into a HDF5 file.

Browse HDF5 file
    Open the :ref:`h5browser` in a new window to browse and import data
    from HDF5 file.

------------

"Edit" menu
-----------

.. image:: /images/shots/i_edit.png

Duplicate
    Create a new image which is identical to the currently selected object.

Remove
    Remove currently selected image.

Delete all
    Delete all images.

Copy metadata
    Copy metadata from currently selected image into clipboard.

Paste metadata
    Paste metadata from clipboard into selected image.

Delete object metadata
    Delete metadata from currently selected image.
    Metadata contains additionnal information such as Region of Interest
    or results of computations

------------

"Operation" menu
----------------

.. image:: /images/shots/i_operation.png

Sum
    Create a new image which is the sum of all selected images:

    .. math::
        z_{M} = \sum_{k=0}^{M-1}{z_{k}}

Average
    Create a new image which is the average of all selected images:

    .. math::
        z_{M} = \dfrac{1}{M}\sum_{k=0}^{M-1}{z_{k}}

Difference
    Create a new image which is the difference of the **two** selected images:

    .. math::
        z_{2} = z_{1} - z_{0}

Product
    Create a new image which is the product of all selected images:

    .. math::
        z_{M} = \prod_{k=0}^{M-1}{z_{k}}

Division
    Create a new image which is the division of the **two** selected images:

    .. math::
        z_{2} = \dfrac{z_{1}}{z_{0}}

Absolute value
    Create a new image which is the absolute value of each selected image:

    .. math::
        z_{k} = |z_{k-1}|

Log10(z)
    Create a new image which is the base 10 logarithm of each selected image:

    .. math::
        z_{k} = \log_{10}(z_{k-1})

Log10(z+n)
    Create a new image which is the Log10(z+n) of each selected image
    (avoid Log10(0) on image background):

    .. math::
        z_{k} = \log_{10}(z_{k-1}+n)

z/√2
    Create a new image which is the result of a division by √2
    of each selected image:

    .. math::
        z_{k} = \dfrac{z_{k-1}}{\sqrt{2}}

Flat-field correction
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
    Create a new image which is the result of rotating (90°, 270° or
    arbitrary angle) or flipping (horizontally or vertically) data.

Resize
    Create a new image which is a resized version of each selected image.

ROI extraction
    Create a new image from a user-defined Region of Interest.

    .. figure:: /images/shots/i_roi_dialog.png

        ROI extraction dialog: the ROI is defined by moving the position
        and adjusting the size of a rectangle shape.

Swap X/Y axes
    Create a new image which is the result of swapping X/Y data.

------------

"Processing" menu
-----------------

.. image:: /images/shots/i_processing.png

Linear calibration
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
    Apply the thresholding to each selected image.

Clipping
    Apply the clipping to each selected image.

Moving average
    Compute moving average of each selected image
    (implementation based on `scipy.ndimage.uniform_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html>`_).

Moving median
    Compute moving median of each selected image
    (implementation based on `scipy.signal.medfilt <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html>`_).

Wiener filter
    Compute Wiener filter of each selected image
    (implementation based on `scipy.signal.wiener <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html>`_).

FFT
    Create a new image which is the Fast Fourier Transform (FFT)
    of each selected image.

Inverse FFT
    Create a new image which is the inverse FFT of each selected image.

------------

"Computing" menu
----------------

.. image:: /images/shots/i_computing.png

Regions of interest
    Open a dialog box to setup multiple Region Of Interests (ROI).
    ROI are stored as metadata, and thus attached to image.

    ROI definition dialog is exactly the same as ROI extraction (see above).

    .. figure:: /images/shots/i_roi_image.png

        An image with ROI.

Statistics
    Compute statistics on selected image and show a summary table.

    .. figure:: /images/shots/i_stats.png

        Example of statistical summary table: each row is associated to an ROI
        (the first row gives the statistics for the whole data).

Centroid
    Compute image centroid using a Fourier transform method
    (as discussed by `Weisshaar et al. <http://www.mnd-umwelttechnik.fh-wiesbaden.de/pig/weisshaar_u5.pdf>`_).
    This method is quite insensitive to background noise.

Minimum enclosing circle center
    Compute the circle contour enclosing image values above
    a threshold level defined as the half-maximum value.

    .. warning::
        This feature requires `OpenCV for Python <https://pypi.org/project/opencv-python/>`_.

2D peak detection
    Automatically find peaks on image using a minimum-maximum filter algorithm.

    .. figure:: /images/shots/i_peak2d_test.png

        Example of 2D peak detection.

Contour detection
    Automatically extract contours and fit them using a circle or an ellipse.

    .. figure:: /images/shots/i_contour_test.png

        Example of contour detection.

.. note:: Computed scalar results are systematically stored as metadata.
    Metadata is attached to image and serialized with it when exporting
    current session in a HDF5 file.

------------

"View" menu
-----------

.. image:: /images/shots/i_view.png
