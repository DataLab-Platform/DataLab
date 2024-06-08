.. _ima-menu-operations:

Operations on Images
====================

This section describes the operations that can be performed on images.

.. seealso::

    :ref:`ima-menu-processing` for more information on image processing features,
    or :ref:`ima-menu-computing` for information on computing features on images.

.. figure:: /images/shots/i_operation.png

    Screenshot of the "Operations" menu.

When the "Image Panel" is selected, the menus and toolbars are updated to
provide image-related actions.

The "Operations" menu allows you to perform various operations on the current
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

Constant operations
^^^^^^^^^^^^^^^^^^^

Create a new image which is the result of a constant operation on each selected image:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Operation
      - Equation
    * - Addition
      - :math:`z_{k} = z_{k-1} + conv(c)`
    * - Subtraction
      - :math:`z_{k} = z_{k-1} - conv(c)`
    * - Multiplication
      - :math:`z_{k} = conv(z_{k-1} \times c)`
    * - Division
      - :math:`z_{k} = conv(\dfrac{z_{k-1}}{c})`

where :math:`c` is the constant value and :math:`conv` is the conversion function
which handles data type conversion (keeping the same data type as the input image).

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

Exponential
^^^^^^^^^^^

Create a new image which is the exponential of each selected image:

.. math::
    z_{k} = \exp(z_{k-1})

Logarithm (base 10)
^^^^^^^^^^^^^^^^^^^

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

Segment profile
    Extract a segment profile from each selected image, and create new signals
    from these profiles.

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
