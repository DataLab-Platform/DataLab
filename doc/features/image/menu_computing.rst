.. _ima-menu-computing:

Computing features on Images
============================

This section describes the image computing features available in DataLab.

.. seealso::

    :ref:`ima-menu-operations` for more information on operations that can be performed
    on images, or :ref:`ima-menu-processing` for information on processing features on
    images.

.. figure:: /images/shots/i_computing.png

    Screenshot of the "Computing" menu.

When the "Image Panel" is selected, the menus and toolbars are updated to
provide image-related actions.

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

Histogram
^^^^^^^^^

Compute histogram of selected image and show it in the Signal Panel.

Parameters are:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Parameter
      - Description
    * - Bins
      - Number of bins
    * - Lower limit
      - Lower limit of the histogram
    * - Upper limit
      - Upper limit of the histogram

.. figure:: /images/shots/i_histogram.png

    Example of histogram.

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
