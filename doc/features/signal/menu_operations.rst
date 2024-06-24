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

Basic arithmetic operations
---------------------------

.. list-table::
    :header-rows: 1
    :widths: 40, 60

    * - Operation
      - Description
    * - |sum| Sum
      - :math:`y_{M} = \sum_{k=0}^{M-1}{y_{k}}`
    * - |average| Average
      - :math:`y_{M} = \dfrac{1}{M}\sum_{k=0}^{M-1}{y_{k}}`
    * - |difference| Difference
      - :math:`y_{2} = y_{1} - y_{0}`
    * - |product| Product
      - :math:`y_{M} = \prod_{k=0}^{M-1}{y_{k}}`
    * - |division| Division
      - :math:`y_{2} = \dfrac{y_{1}}{y_{0}}`

.. |sum| image:: ../../../cdl/data/icons/operations/sum.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |average| image:: ../../../cdl/data/icons/operations/average.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |difference| image:: ../../../cdl/data/icons/operations/difference.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |product| image:: ../../../cdl/data/icons/operations/product.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |division| image:: ../../../cdl/data/icons/operations/division.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

Operations with a constant
--------------------------

Create a new signal which is the result of a constant operation on each selected signal:

.. list-table::
    :header-rows: 1
    :widths: 40, 60

    * - Operation
      - Description
    * - |constant_add| Addition
      - :math:`y_{k} = y_{k-1} + c`
    * - |constant_substract| Subtraction
      - :math:`y_{k} = y_{k-1} - c`
    * - |constant_multiply| Multiplication
      - :math:`y_{k} = y_{k-1} \times c`
    * - |constant_divide| Division
      - :math:`y_{k} = \dfrac{y_{k-1}}{c}`

.. |constant_add| image:: ../../../cdl/data/icons/operations/constant_add.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |constant_substract| image:: ../../../cdl/data/icons/operations/constant_substract.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |constant_multiply| image:: ../../../cdl/data/icons/operations/constant_multiply.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |constant_divide| image:: ../../../cdl/data/icons/operations/constant_divide.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

Real and imaginary parts
------------------------

.. list-table::
    :header-rows: 1
    :widths: 40, 60

    * - Operation
      - Description
    * - |abs| Absolute value
      - :math:`y_{k} = |y_{k-1}|`
    * - |re| Real part
      - :math:`y_{k} = \Re(y_{k-1})`
    * - |im| Imaginary part
      - :math:`y_{k} = \Im(y_{k-1})`

.. |abs| image:: ../../../cdl/data/icons/operations/abs.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |re| image:: ../../../cdl/data/icons/operations/re.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |im| image:: ../../../cdl/data/icons/operations/im.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

Data type conversion
--------------------

The "Convert data type" |convert_dtype| action allows you to convert the data type
of the selected signals.

.. |convert_dtype| image:: ../../../cdl/data/icons/operations/convert_dtype.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. note::

    Data type conversion relies on :py:func:`numpy.ndarray.astype` function with
    the default parameters (`casting='unsafe'`).

Basic mathematical functions
----------------------------

.. list-table::
    :header-rows: 1
    :widths: 40, 60

    * - Function
      - Description
    * - |exp| Exponential
      - :math:`y_{k} = \exp(y_{k-1})`
    * - |log10| Logarithm (base 10)
      - :math:`y_{k} = \log_{10}(y_{k-1})`
    * - |power| Power
      - :math:`y_{k} = y_{k-1}^{n}`
    * - |sqrt| Square root
      - :math:`y_{k} = \sqrt{y_{k-1}}`

.. |exp| image:: ../../../cdl/data/icons/operations/exp.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |log10| image:: ../../../cdl/data/icons/operations/log10.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |power| image:: ../../../cdl/data/icons/operations/power.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |sqrt| image:: ../../../cdl/data/icons/operations/sqrt.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

Other mathematical operations
-----------------------------

.. list-table::
    :header-rows: 1
    :widths: 40, 60

    * - Operation
      - Implementation
    * - |derivative| Derivative
      - Based on `numpy.gradient <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_
    * - |integral| Integral
      - Based on `scipy.integrate.cumulative_trapezoid <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html>`_
    * - |convolution| Convolution
      - Based on `scipy.signal.convolve <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html>`_

.. |derivative| image:: ../../../cdl/data/icons/operations/derivative.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |integral| image:: ../../../cdl/data/icons/operations/integral.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link

.. |convolution| image:: ../../../cdl/data/icons/operations/convolution.svg
    :width: 24px
    :height: 24px
    :class: dark-light no-scaled-link
