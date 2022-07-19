"Processing" menu
=================

.. image:: /images/shots/s_processing.png

Normalize
    Create a new signal which is the normalization of each selected signal
    by maximum, amplitude, sum or energy:

    .. list-table::
        :header-rows: 1
        :widths: 25, 75

        * - Parameter
          - Normalization
        * - Maximum
          - :math:`y_{1}= \dfrac{y_{0}}{max(y_{0})}`
        * - Amplitude
          - :math:`y_{1}= \dfrac{y_{0}'}{max(y_{0}')}` with :math:`y_{0}'=y_{0}-min(y_{0})`
        * - Sum
          - :math:`y_{1}= \dfrac{y_{0}}{\sum_{n=0}^{N}y_{0}[n]}`
        * - Energy
          - :math:`y_{1}= \dfrac{y_{0}}{\sum_{n=0}^{N}|y_{0}[n]|^2}`

Derivative
    Create a new signal which is the derivative of each selected signal.

Integral
    Create a new signal which is the integral of each selected signal.

Linear calibration
    Create a new signal which is a linear calibration of each selected signal
    with respect to X or Y axis:

    .. list-table::
        :header-rows: 1
        :widths: 40, 60

        * - Parameter
          - Linear calibration
        * - X-axis
          - :math:`x_{1} = a.x_{0} + b`
        * - Y-axis
          - :math:`y_{1} = a.y_{0} + b`

Gaussian filter
    Compute 1D-Gaussian filter of each selected signal
    (implementation based on `scipy.ndimage.gaussian_filter1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html>`_).

Moving average
    Compute moving average on :math:`M`
    points of each selected signal, without border effect:

    .. math::
        y_{1}[i]=\dfrac{1}{M}\sum_{j=0}^{M-1}y_{0}[i+j]

Moving median
    Compute moving median of each selected signal
    (implementation based on `scipy.signal.medfilt <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html>`_).

Wiener filter
    Compute Wiener filter of each selected signal
    (implementation based on `scipy.signal.wiener <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html>`_).

FFT
    Create a new signal which is the Fast Fourier Transform (FFT)
    of each selected signal.

Inverse FFT
    Create a new signal which is the inverse FFT of each selected signal.

Lorentzian, Lorentzian, Voigt, Polynomial and Multi-Gaussian fit
    Open an interactive curve fitting tool in a modal dialog box.

    .. list-table::
        :header-rows: 1
        :widths: 20, 80

        * - Model
          - Equation
        * - Gaussian
          - :math:`y = y_{0}+\dfrac{A}{\sqrt{2\pi}.\sigma}.exp(-\dfrac{1}{2}.(\dfrac{x-x_{0}}{\sigma})^2)`
        * - Lorentzian
          - :math:`y = y_{0}+\dfrac{A}{\sigma.\pi}.\dfrac{1}{1+(\dfrac{x-x_{0}}{\sigma})^2}`
        * - Voigt
          - :math:`y = y_{0}+A.\dfrac{Re(exp(-z^2).erfc(-j.z))}{\sqrt{2\pi}.\sigma}` with :math:`z = \dfrac{x-x_{0}-j.\sigma}{\sqrt{2}.\sigma}`
        * - Multi-Gaussian
          - :math:`y = y_{0}+\sum_{i=0}^{K}\dfrac{A_{i}}{\sqrt{2\pi}.\sigma_{i}}.exp(-\dfrac{1}{2}.(\dfrac{x-x_{0,i}}{\sigma_{i}})^2)`
