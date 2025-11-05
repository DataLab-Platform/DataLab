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
  - Optionally, enable "Create regions of interest" to automatically create
    ROIs around each detected peak:

    * Choose ROI geometry: "Rectangle" or "Circle"
    * ROI size is automatically calculated based on the minimum distance
      between detected peaks (to avoid overlap)
    * This feature requires at least 2 detected peaks
    * Created ROIs can be useful for subsequent processing on each peak area
      (e.g., contour detection, measurements, etc.)

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
  - "Create regions of interest": if enabled, automatically creates ROIs around
    each detected peak (requires at least 2 peaks)
  - "ROI geometry": shape of the ROIs ("Rectangle" or "Circle")

Feature is based on ``get_2d_peaks_coords`` function from ``sigima.tools`` module:

  .. literalinclude:: ../../../../Sigima/sigima/tools/image/detection.py
     :pyobject: get_2d_peaks_coords
