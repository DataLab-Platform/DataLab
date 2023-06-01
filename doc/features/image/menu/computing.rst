"Computing" menu
================

.. image:: /images/shots/i_computing.png

Edit regions of interest
    Open a dialog box to setup multiple Region Of Interests (ROI).
    ROI are stored as metadata, and thus attached to image.

    ROI definition dialog is exactly the same as ROI extraction (see above).

    .. figure:: /images/shots/i_roi_image.png

        An image with ROI.

Remove regions of interest
    Remove all defined ROI for selected object(s).

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

2D peak detection
    Automatically find peaks on image using a minimum-maximum filter algorithm.

    .. figure:: /images/shots/i_peak2d_test.png

        Example of 2D peak detection.

    .. seealso::
        See :ref:`ref-to-2d-peak-detection` for more details on algorithm and associated parameters.

Contour detection
    Automatically extract contours and fit them using a circle or an ellipse.

    .. figure:: /images/shots/i_contour_test.png

        Example of contour detection.

    .. seealso::
        See :ref:`ref-to-contour-detection` for more details on algorithm and associated parameters.

.. note:: Computed scalar results are systematically stored as metadata.
    Metadata is attached to image and serialized with it when exporting
    current session in a HDF5 file.

Circle Hough transform
    Detect circular shapes using circle Hough transform
    (implementation based on `skimage.transform.hough_circle_peaks <https://scikit-image.org/docs/stable/api/skimage.transform.html?highlight=hough#skimage.transform.hough_circle_peaks>`_).

Blob detection
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
