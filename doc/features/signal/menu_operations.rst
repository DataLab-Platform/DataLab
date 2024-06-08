.. _sig-menu-operations:

Operations on Signals
=====================

This section describes the operations that can be performed on signals.

.. seealso::

    :ref:`sig-menu-processing` for more information on signal processing features,
    or :ref:`sig-menu-computing` for information on computing features on signals.

.. figure:: /images/shots/s_operation.png

    Screenshot of the "Operations" menu.

When the "Signal Panel" is selected, the menus and toolbars are updated to
provide signal-related actions.

The "Operations" menu allows you to perform various operations on the
selected signals, such as arithmetic operations, peak detection, or
convolution.

Sum
^^^

Create a new signal which is the sum of all selected signals:

.. math::
    y_{M} = \sum_{k=0}^{M-1}{y_{k}}

Average
^^^^^^^

Create a new signal which is the average of all selected signals:

.. math::
    y_{M} = \dfrac{1}{M}\sum_{k=0}^{M-1}{y_{k}}

Difference
^^^^^^^^^^

Create a new signal which is the difference of the **two** selected signals:

.. math::
    y_{2} = y_{1} - y_{0}

Product
^^^^^^^

Create a new signal which is the product of all selected signals:

.. math::
    y_{M} = \prod_{k=0}^{M-1}{y_{k}}

Division
^^^^^^^^

Create a new signal which is the division of the **two** selected signals:

.. math::
    y_{2} = \dfrac{y_{1}}{y_{0}}

Constant operations
^^^^^^^^^^^^^^^^^^^

Create a new signal which is the result of a constant operation on each selected signal:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Operation
      - Description
    * - Addition
      - :math:`y_{k} = y_{k-1} + c`
    * - Subtraction
      - :math:`y_{k} = y_{k-1} - c`
    * - Multiplication
      - :math:`y_{k} = y_{k-1} \times c`
    * - Division
      - :math:`y_{k} = \dfrac{y_{k-1}}{c}`

Absolute value
^^^^^^^^^^^^^^

Create a new signal which is the absolute value of each selected signal:

.. math::
    y_{k} = |y_{k-1}|

Real part
^^^^^^^^^

Create a new signal which is the real part of each selected signal:

.. math::
    y_{k} = \Re(y_{k-1})

Imaginary part
^^^^^^^^^^^^^^

Create a new signal which is the imaginary part of each selected signal:

.. math::
    y_{k} = \Im(y_{k-1})

Convert data type
^^^^^^^^^^^^^^^^^

Create a new signal which is the result of converting data type of each selected signal.

.. note::

    Data type conversion relies on :py:func:`numpy.ndarray.astype` function with
    the default parameters (`casting='unsafe'`).

Exponential
^^^^^^^^^^^

Create a new signal which is the exponential of each selected signal:

.. math::
    y_{k} = \exp(y_{k-1})

Logarithm (base 10)
^^^^^^^^^^^^^^^^^^^

Create a new signal which is the base 10 logarithm of each selected signal:

.. math::
    y_{k} = \log_{10}(y_{k-1})

Power
^^^^^

Create a new signal which is the power of each selected signal:

.. math::
    y_{k} = y_{k-1}^{n}

Square root
^^^^^^^^^^^

Create a new signal which is the square root of each selected signal:

.. math::
    y_{k} = \sqrt{y_{k-1}}

Derivative
^^^^^^^^^^

Create a new signal which is the derivative of each selected signal.

Integral
^^^^^^^^

Create a new signal which is the integral of each selected signal.

Convolution
^^^^^^^^^^^

Create a new signal which is the convolution of each selected signal
with respect to another signal.

This feature is based on SciPy's `scipy.signal.convolve <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html>`_ function.
