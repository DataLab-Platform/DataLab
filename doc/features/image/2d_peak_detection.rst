.. _ref-to-2d-peak-detection:

2D Peak Detection
=================

DataLab provides a "2D Peak Detection" feature which is based on a
minimum-maximum filter algorithm.

.. figure:: /images/2d_peak_detection/peak2d_app_param.png

    2D peak detection parameters.

How to use the feature:
  - Create or open an image in DataLab workspace
  - Select "2d peak detection" in "Analysis" menu
  - Enter parameters "Neighborhoods size" and "Relative threhold"
  - Check "Create regions of interest" if you want a ROI defined for each
    detected peak (this may become useful when using another computation
    afterwards on each area around peaks, e.g. contour detection)

.. figure:: /images/2d_peak_detection/peak2d_app_results.png

    2d peak detection results (see test "peak2d_app.py")

Results are shown in a table:
  - Each row is associated to a detected peak
  - First column shows the ROI index (0 if no ROI is defined on input image)
  - Second and third columns show peak coordinates

.. figure:: /images/2d_peak_detection/peak2d_app_zoom.png

    Example of 2D peak detection.

The 2d peak detection algorithm works in the following way:
  - First, the minimum and maximum filtered images are computed
    using a sliding window algorithm with a user-defined size
    (implementation based on `scipy.ndimage.minimum_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.minimum_filter.html>`_
    and `scipy.ndimage.maximum_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter.html>`_)
  - Then, the difference between the maximum and minimum filtered
    images is clipped at a user-defined threshold
  - Resulting image features are labeled using `scipy.ndimage.label <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html>`_
  - Peak coordinates are then obtained from labels center
  - Duplicates are eventually removed

The 2d peak detection parameters are the following:
  - "Neighborhoods size": size of the sliding window (see above)
  - "Relative threshold": detection threshold

Feature is based on ``get_2d_peaks_coords`` function
from ``cdl.algorithms`` module:

  .. literalinclude:: ../../../cdl/algorithms/image.py
     :pyobject: get_2d_peaks_coords
