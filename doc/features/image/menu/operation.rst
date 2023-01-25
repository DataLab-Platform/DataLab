"Operation" menu
================

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

Quadratic difference
    Create a new image which is the quadratic difference of the **two**
    selected images:

    .. math::
        z_{2} = \dfrac{z_{1} - z_{0}}{\sqrt{2}}

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

Pixel binning
    Combine clusters of adjacent pixels, throughout the image,
    into single pixels. The result can be the sum, average, median, minimum,
    or maximum value of the cluster.

ROI extraction
    Create a new image from a user-defined Region of Interest.

    .. figure:: /images/shots/i_roi_dialog.png

        ROI extraction dialog: the ROI is defined by moving the position
        and adjusting the size of a rectangle shape.

Swap X/Y axes
    Create a new image which is the result of swapping X/Y data.
