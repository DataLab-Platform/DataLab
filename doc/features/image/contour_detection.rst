.. _ref-to-contour-detection:

Contour Detection
=================

DataLab provides a "Contour Detection" feature which is based on
`the marching cubes algorithm <https://courses.cs.duke.edu/fall01/cps124/resources/p163-lorensen.pdf>`_.

.. figure:: /images/contour_detection/contour_app_param.png

    Contour detection parameters.

How to use the feature:
  - Create or open an image in DataLab workspace
  - Eventually create a ROI around the target area
  - Select "Contour detection" in "Analysis" menu
  - Enter parameter "Shape" ("Ellipse", "Circle" or "Polygon")

.. figure:: /images/contour_detection/contour_app_results.png

    Contour detection results (see test "contour_app.py")

Results are shown in a table:
  - Each row is associated to a contour
  - First column shows the ROI index (0 if no ROI is defined on input image)
  - Other columns show contour coordinates:
    4 columns for circles (coordinates of diameter),
    8 columns for ellipses (coordinates of diameters)

.. figure:: /images/contour_detection/contour_app.png

    Example of contour detection.

The contour detection algorithm works in the following way:
  - First, iso-valued contours are computed
    (implementation based on `skimage.measure.find_contours.find_contours <https://scikit-image.org/docs/0.8.0/api/skimage.measure.find_contours.html#find-contours>`_)
  - Then, each contour is fitted to the closest ellipse (or circle)

Feature is based on ``get_contour_shapes`` function
from ``cdl.algorithms`` module:

  .. literalinclude:: ../../../cdl/algorithms/image.py
     :pyobject: get_contour_shapes
