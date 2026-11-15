.. _ref-to-contour-detection:

Contour Detection
=================

DataLab provides a "Contour Detection" feature which is based on
`the marching squares algorithm <https://en.wikipedia.org/wiki/Marching_squares>`_.

.. figure:: /images/contour_detection/contour_app_param.png

    Contour detection parameters.

How to use the feature:
  - Create or open an image in DataLab workspace
  - Eventually create a ROI around the target area
  - Select "Contour detection" in "Analysis" menu
  - Enter parameters "Relative threshold" and "Shape" ("Ellipse", "Circle" or "Polygon")
  - Optionally, enable "Create regions of interest" to automatically create
    a ROI from each detected contour:

    * ROI geometry is determined by the selected shape: circular ROIs for
      circles, polygonal ROIs for polygons and ellipses (approximated by a polygon)
    * Created ROIs can be useful for subsequent processing on each contour area

.. figure:: /images/contour_detection/contour_app_results.png

    Contour detection results (see test "contour_app_test.py")

Results are shown in a table:
  - Each row is associated to a contour, and is labelled with the name of the
    function and the reference of the image (followed by the ROI title, if a ROI
    is defined on input image)
  - Columns show contour coordinates:
    3 columns for circles (center coordinates x, y and radius r),
    5 columns for ellipses (center coordinates x, y, semi-axes a, b and angle θ),
    and vertex coordinates (x0, y0, x1, y1, ...) for polygons

.. figure:: /images/contour_detection/contour_app.png

    Example of contour detection.

The contour detection algorithm works in the following way:
  - First, iso-valued contours are computed
    (implementation based on `skimage.measure.find_contours.find_contours <https://scikit-image.org/docs/0.8.0/api/skimage.measure.find_contours.html#find-contours>`_)
  - Then, each contour is fitted to the closest ellipse (or circle)

Feature is based on ``get_contour_shapes`` function from ``sigima.tools`` module:

  .. literalinclude:: ../../../../Sigima/sigima/tools/image/detection.py
     :pyobject: get_contour_shapes
