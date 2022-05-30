Signal processing
=================

.. figure:: /images/shots/s_simple_example.png

    CodraFT main window: Signal processing view


"File" menu
-----------

.. image:: /images/shots/s_file.png

New signal
    Create a new signal from various models:

    .. list-table::
        :header-rows: 1
        :widths: 20, 80

        * - Model
          - Equation
        * - Zeros
          - :math:`y[i] = 0`
        * - Random
          - :math:`y[i] \in [-0.5, 0.5]`
        * - Gaussian
          - :math:`y = y_{0}+\dfrac{A}{\sqrt{2\pi}.\sigma}.exp(-\dfrac{1}{2}.(\dfrac{x-x_{0}}{\sigma})^2)`
        * - Lorentzian
          - :math:`y = y_{0}+\dfrac{A}{\sigma.\pi}.\dfrac{1}{1+(\dfrac{x-x_{0}}{\sigma})^2}`
        * - Voigt
          - :math:`y = y_{0}+A.\dfrac{Re(exp(-z^2).erfc(-j.z))}{\sqrt{2\pi}.\sigma}` with :math:`z = \dfrac{x-x_{0}-j.\sigma}{\sqrt{2}.\sigma}`

Open signal
    Create a new signal from the following supported filetypes:

    .. list-table::
        :header-rows: 1

        * - File type
          - Extensions
        * - Text files
          - .txt, .csv
        * - NumPy arrays
          - .npy

Save signal
    Save current signal to the following supported filetypes:

    .. list-table::
        :header-rows: 1

        * - File type
          - Extensions
        * - Text files
          - .csv

Import metadata into signal
    Import metadata from a JSON text file.

Export metadata from signal
    Export metadata to a JSON text file.

Open HDF5 file
    Import data from a HDF5 file.

Save to HDF5 file
    Export the whole CodraFT session (all signals and images) into a HDF5 file.

Browse HDF5 file
    Open the :ref:`h5browser` in a new window to browse and import data
    from HDF5 file.

------------

"Edit" menu
-----------

.. image:: /images/shots/s_edit.png

Duplicate
    Create a new signal which is identical to the currently selected object.

Remove
    Remove currently selected signal.

Delete all
    Delete all signals.

Copy metadata
    Copy metadata from currently selected image into clipboard.

Paste metadata
    Paste metadata from clipboard into selected image.

Delete object metadata
    Delete metadata from currently selected signal.
    Metadata contains additionnal information such as Region of Interest
    or results of computations

------------

"Operation" menu
----------------

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

ROI extraction
    Create a new signal from a user-defined Region of Interest (ROI).

    .. figure:: /images/shots/s_roi_dialog.png

        ROI extraction dialog: the ROI is defined by moving the position
        and adjusting the width of an horizontal range.

Swap X/Y axes
    Create a new signal which is the result of swapping X/Y data.

------------

"Processing" menu
-----------------

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

Lorentzian filter
    Compute 1D-Lorentzian filter of each selected signal
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

------------

"Computing" menu
----------------

.. image:: /images/shots/s_computing.png

Regions of interest
    Open a dialog box to setup multiple Region Of Interests (ROI).
    ROI are stored as metadata, and thus attached to signal.

    ROI definition dialog is exactly the same as ROI extraction (see above):
    the ROI is defined by moving the position and adjusting the width of an
    horizontal range.

    .. figure:: /images/shots/s_roi_signal.png

        A signal with an ROI.

Statistics
    Compute statistics on selected signal and show a summary table.

    .. figure:: /images/shots/s_stats.png

        Example of statistical summary table: each row is associated to an ROI
        (the first row gives the statistics for the whole data).

Full width at half-maximum
    Fit data to a Gaussian, Lorentzian or Voigt model using
    least-square method.
    Then, compute the full width at half-maximum value.

    .. figure:: /images/shots/s_fwhm.png

        The computed result is displayed as an annotated segment.

Full width at 1/e²
    Fit data to a Gaussian model using least-square method.
    Then, compute the full width at 1/e².

.. note:: Computed scalar results are systematically stored as metadata.
    Metadata is attached to signal and serialized with it when exporting
    current session in a HDF5 file.

------------

"View" menu
-----------

.. image:: /images/shots/s_view.png

View in a new window
    Open a new window to visualize the selected signals.

Other menu entries
    Show/hide panels or toolbars.

------------

"?" menu
--------

.. image:: /images/shots/s_help.png

Online documentation
    Open the online documentation (english only):

    .. image:: /images/shots/doc_online.png

CHM documentation
    Open the CHM documentation (french/english and Windows only):

    .. image:: /images/shots/doc_chm.png

About
    Open the "About CodraFT" dialog box:

    .. image:: /images/shots/about.png
