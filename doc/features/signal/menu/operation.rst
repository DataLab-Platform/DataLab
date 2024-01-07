"Operation" menu
================

.. image:: /images/shots/s_operation.png

Sum
    Create a new signal which is the sum of all selected signals:

    .. math::
        y_{M} = \sum_{k=0}^{M-1}{y_{k}}

Average
    Create a new signal which is the average of all selected signals:

    .. math::
        y_{M} = \dfrac{1}{M}\sum_{k=0}^{M-1}{y_{k}}

Difference
    Create a new signal which is the difference of the **two** selected
    signals:

    .. math::
        y_{2} = y_{1} - y_{0}

Product
    Create a new signal which is the product of all selected signals:

    .. math::
        y_{M} = \prod_{k=0}^{M-1}{y_{k}}

Division
    Create a new signal which is the division of the **two** selected signals:

    .. math::
        y_{2} = \dfrac{y_{1}}{y_{0}}

Absolute value
    Create a new signal which is the absolute value of each selected signal:

    .. math::
        y_{k} = |y_{k-1}|

Real part
    Create a new signal which is the real part of each selected signal:

    .. math::
        y_{k} = \Re(y_{k-1})

Imaginary part
    Create a new signal which is the imaginary part of each selected signal:

    .. math::
        y_{k} = \Im(y_{k-1})

Convert data type
    Create a new signal which is the result of converting data type of each
    selected signal.

    .. note::

        Data type conversion relies on :py:func:`numpy.ndarray.astype` function with
        the default parameters (`casting='unsafe'`).

Log10(y)
    Create a new signal which is the base 10 logarithm of each selected signal:

    .. math::
        z_{k} = \log_{10}(z_{k-1})

Peak detection
    Create a new signal from semi-automatic peak detection of each selected
    signal.

    .. figure:: /images/shots/s_peak_detection.png

        Peak detection dialog: threshold is adjustable by moving the
        horizontal marker, peaks are detected automatically (see vertical
        markers with labels indicating peak position)

Convolution
    Create a new signal which is the convolution of each selected signal
    with respect to another signal.

    This feature is based on SciPy's `scipy.signal.convolve <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html>`_ function.

ROI extraction
    Create a new signal from a user-defined Region of Interest (ROI).

    .. figure:: /images/shots/s_roi_dialog.png

        ROI extraction dialog: the ROI is defined by moving the position
        and adjusting the width of an horizontal range.

Swap X/Y axes
    Create a new signal which is the result of swapping X/Y data.
