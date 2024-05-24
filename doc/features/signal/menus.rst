Menus
=====

This section describes the signal related feature of DataLab, by presenting
the different menus and their entries.

"File" menu
-----------

.. figure:: /images/shots/s_file.png

    Screenshot of the "File" menu.

The "File" menu allows you to create, open, save and close signals. It also
allows you to import and export data from/to HDF5 files, and to edit the
settings of the current session.

New signal
^^^^^^^^^^

Create a new signal from various models:

.. list-table::
    :header-rows: 1
    :widths: 20, 80

    * - Model
      - Equation
    * - Zeros
      - :math:`y[i] = 0`
    * - Gaussian
      - :math:`y = y_{0}+\dfrac{A}{\sqrt{2\pi}.\sigma}.exp(-\dfrac{1}{2}.(\dfrac{x-x_{0}}{\sigma})^2)`
    * - Lorentzian
      - :math:`y = y_{0}+\dfrac{A}{\sigma.\pi}.\dfrac{1}{1+(\dfrac{x-x_{0}}{\sigma})^2}`
    * - Voigt
      - :math:`y = y_{0}+A.\dfrac{Re(exp(-z^2).erfc(-j.z))}{\sqrt{2\pi}.\sigma}` with :math:`z = \dfrac{x-x_{0}-j.\sigma}{\sqrt{2}.\sigma}`
    * - Random (uniform law)
      - :math:`y[i] \in [-0.5, 0.5]`
    * - Random (normal law)
      - :math:`y[i] \sim \mathcal{N}(-0.5, 0.5)`
    * - Sine
      - :math:`y = y_{0}+A.sin(2\pi.f.x+\phi)`
    * - Cosine
      - :math:`y = y_{0}+A.cos(2\pi.f.x+\phi)`
    * - Sawtooth
      - :math:`y = y_{0}+A \cdot \left( 2 \left( f x + \frac{\phi}{2\pi} - \left\lfloor f x + \frac{\phi}{2\pi} + \frac{1}{2} \right\rfloor \right) \right)`
    * - Triangle
      - :math:`y = y_{0}+A \cdot \text{sawtooth}(2 \pi f x + \phi, \text{width} = 0.5)`
    * - Square
      - :math:`y = y_0 + A \cdot \text{sgn}\left( \sin\left( 2\pi f x + \phi \right) \right)`
    * - Cardinal sine
      - :math:`y = y_0 + A \cdot \text{sinc}\left(2\pi f x + \phi\right)`
    * - Step
      - :math:`y = y_{0}+A.\left\{\begin{array}{ll}1 & \text{if } x > x_{0} \\ 0 & \text{otherwise}\end{array}\right.`
    * - Exponential
      - :math:`y = y_{0}+A.exp(B.x)`
    * - Pulse
      - :math:`y = y_{0}+A.\left\{\begin{array}{ll}1 & \text{if } x_{0} < x < x_{1} \\ 0 & \text{otherwise}\end{array}\right.`
    * - Polynomial
      - :math:`y = y_{0}+A_{0}+A_{1}.x+A_{2}.x^2+...+A_{n}.x^n`
    * - Experimental
      - Manual input of X and Y values

.. _open_signal:

Open signal
^^^^^^^^^^^

Create a new signal from the following supported filetypes:

.. list-table::
    :header-rows: 1

    * - File type
      - Extensions
    * - Text files
      - .txt, .csv
    * - NumPy arrays
      - .npy
    * - MAT-Files
      - .mat

Save signal
^^^^^^^^^^^

Save current signal to the following supported filetypes:

.. list-table::
    :header-rows: 1

    * - File type
      - Extensions
    * - Text files
      - .csv

Import text file
^^^^^^^^^^^^^^^^

Import data from a text file.

.. seealso::

    See :ref:`ref-to-signal-text-file-import` page for more details on importing text files.

Open HDF5 file
^^^^^^^^^^^^^^

Import data from a HDF5 file.

Save to HDF5 file
^^^^^^^^^^^^^^^^^

Export the whole DataLab session (all signals and images) into a HDF5 file.

Browse HDF5 file
^^^^^^^^^^^^^^^^

Open the :ref:`h5browser` in a new window to browse and import data from HDF5 file.

Settings
^^^^^^^^

Open the the "Settings" dialog box.

.. image:: /images/settings.png

"Edit" menu
-----------

.. figure:: /images/shots/s_edit.png

    Screenshot of the "Edit" menu.

The "Edit" menu allows you to edit the current signal or group of signals, by
adding, removing, renaming, moving up or down, or duplicating signals. It also
manipulates metadata, or handles signal titles.

New group
^^^^^^^^^

Create a new group of signals. Images may be moved from one group to another
by drag and drop.

Rename group
^^^^^^^^^^^^

Rename currently selected group.

Move up
^^^^^^^

Move current selection up in the list (groups or signals may be selected). If
multiple objects are selected, they are moved together. If a selected signal
is already at the top of its group, it is moved to the bottom of the previous
group.

Move down
^^^^^^^^^

Move current selection down in the list (groups or signals may be selected). If
multiple objects are selected, they are moved together. If a selected signal
is already at the bottom of its group, it is moved to the top of the next
group.

Duplicate
^^^^^^^^^

Create a new signal which is identical to the currently selected object.

Remove
^^^^^^

Remove currently selected signal.

Delete all
^^^^^^^^^^

Delete all signals.

Copy metadata
^^^^^^^^^^^^^

Copy metadata from currently selected signal into clipboard.

Paste metadata
^^^^^^^^^^^^^^

Paste metadata from clipboard into selected signal.

Import metadata into signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Import metadata from a JSON text file.

Export metadata from signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Export metadata to a JSON text file.

Delete object metadata
^^^^^^^^^^^^^^^^^^^^^^

Delete metadata from currently selected signal.
Metadata contains additionnal information such as Region of Interest
or results of computations

Add object title to plot
^^^^^^^^^^^^^^^^^^^^^^^^

Add currently selected signal title to the associated plot.

Copy titles to clipboard
^^^^^^^^^^^^^^^^^^^^^^^^

Copy all signal titles to clipboard as a multiline text.
This text may be used for reproducing a processing chain, for example.

"Operation" menu
----------------

.. figure:: /images/shots/s_operation.png

    Screenshot of the "Operation" menu.

The "Operation" menu allows you to perform various operations on the
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

Division
^^^^^^^^

Create a new signal which is the division of the **two** selected signals:

.. math::
    y_{2} = \dfrac{y_{1}}{y_{0}}

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

Peak detection
^^^^^^^^^^^^^^

Create a new signal from semi-automatic peak detection of each selected signal.

.. figure:: /images/shots/s_peak_detection.png

    Peak detection dialog: threshold is adjustable by moving the
    horizontal marker, peaks are detected automatically (see vertical
    markers with labels indicating peak position)

Convolution
^^^^^^^^^^^

Create a new signal which is the convolution of each selected signal
with respect to another signal.

This feature is based on SciPy's `scipy.signal.convolve <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html>`_ function.

Windowing
^^^^^^^^^

Create a new signal which is the result of applying a window function to each selected signal.

The following window functions are available:

.. list-table::
    :header-rows: 1
    :widths: 20, 80

    * - Window function
      - Reference
    * - Barthann
      - :py:func:`scipy.signal.windows.barthann`
    * - Bartlett
      - :py:func:`numpy.bartlett`
    * - Blackman
      - :py:func:`scipy.signal.windows.blackman`
    * - Blackman-Harris
      - :py:func:`scipy.signal.windows.blackmanharris`
    * - Bohman
      - :py:func:`scipy.signal.windows.bohman`
    * - Boxcar
      - :py:func:`scipy.signal.windows.boxcar`
    * - Cosine
      - :py:func:`scipy.signal.windows.cosine`
    * - Exponential
      - :py:func:`scipy.signal.windows.exponential`
    * - Flat top
      - :py:func:`scipy.signal.windows.flattop`
    * - Gaussian
      - :py:func:`scipy.signal.windows.gaussian`
    * - Hamming
      - :py:func:`numpy.hamming`
    * - Hanning
      - :py:func:`numpy.hanning`
    * - Kaiser
      - :py:func:`scipy.signal.windows.kaiser`
    * - Lanczos
      - :py:func:`scipy.signal.windows.lanczos`
    * - Nuttall
      - :py:func:`scipy.signal.windows.nuttall`
    * - Parzen
      - :py:func:`scipy.signal.windows.parzen`
    * - Rectangular
      - :py:func:`numpy.ones`
    * - Taylor
      - :py:func:`scipy.signal.windows.taylor`
    * - Tukey
      - :py:func:`scipy.signal.windows.tukey`

ROI extraction
^^^^^^^^^^^^^^

Create a new signal from a user-defined Region of Interest (ROI).

.. figure:: /images/shots/s_roi_dialog.png

    ROI extraction dialog: the ROI is defined by moving the position
    and adjusting the width of an horizontal range.

Swap X/Y axes
^^^^^^^^^^^^^

Create a new signal which is the result of swapping X/Y data.

"Processing" menu
-----------------

.. figure:: /images/shots/s_processing.png

    Screenshot of the "Processing" menu.

The "Processing" menu allows you to perform various processing on the
selected signals, such as smoothing, normalization, or interpolation.

Derivative
^^^^^^^^^^

Create a new signal which is the derivative of each selected signal.

Integral
^^^^^^^^

Create a new signal which is the integral of each selected signal.

Normalize
^^^^^^^^^

Create a new signal which is the normalization of each selected signal
by maximum, amplitude, sum, energy or RMS:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Parameter
      - Normalization
    * - Maximum
      - :math:`y_{1}= \dfrac{y_{0}}{max(y_{0})}`
    * - Amplitude
      - :math:`y_{1}= \dfrac{y_{0}'}{max(y_{0}')}` with :math:`y_{0}'=y_{0}-min(y_{0})`
    * - Area
      - :math:`y_{1}= \dfrac{y_{0}}{\sum_{n=0}^{N}y_{0}[n]}`
    * - Energy
      - :math:`y_{1}= \dfrac{y_{0}}{\sqrt{\sum_{n=0}^{N}|y_{0}[n]|^2}}`
    * - RMS
      - :math:`y_{1}= \dfrac{y_{0}}{\sqrt{\dfrac{1}{N}\sum_{n=0}^{N}|y_{0}[n]|^2}}`

Linear calibration
^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^

Compute 1D-Gaussian filter of each selected signal
(implementation based on `scipy.ndimage.gaussian_filter1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html>`_).

Moving average
^^^^^^^^^^^^^^

Compute moving average on :math:`M`
points of each selected signal, without border effect:

.. math::
    y_{1}[i]=\dfrac{1}{M}\sum_{j=0}^{M-1}y_{0}[i+j]

Moving median
^^^^^^^^^^^^^

Compute moving median of each selected signal
(implementation based on `scipy.signal.medfilt <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html>`_).

Wiener filter
^^^^^^^^^^^^^

Compute Wiener filter of each selected signal
(implementation based on `scipy.signal.wiener <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html>`_).

FFT
^^^

Create a new signal which is the Fast Fourier Transform (FFT) of each selected signal.

Inverse FFT
^^^^^^^^^^^

Create a new signal which is the inverse FFT of each selected signal.

Frequency filters
^^^^^^^^^^^^^^^^^

Create a new signal which is the result of applying a frequency filter to each selected signal.

The following filters are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Filter
      - Description
    * - |lowpass| Low-pass
      - Filter out high frequencies, above a cutoff frequency
    * - |highpass| High-pass
      - Filter out low frequencies, below a cutoff frequency
    * - |bandpass| Band-pass
      - Filter out frequencies outside a range
    * - |bandstop| Band-stop
      - Filter out frequencies inside a range

.. |lowpass| image:: ../../../cdl/data/icons/processing/lowpass.svg
    :width: 24px
    :height: 24px

.. |highpass| image:: ../../../cdl/data/icons/processing/highpass.svg
    :width: 24px
    :height: 24px

.. |bandpass| image:: ../../../cdl/data/icons/processing/bandpass.svg
    :width: 24px
    :height: 24px

.. |bandstop| image:: ../../../cdl/data/icons/processing/bandstop.svg
    :width: 24px
    :height: 24px

For each filter, the following methods are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Method
      - Description
    * - Bessel
      - Bessel filter, using SciPy's `scipy.signal.bessel <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html>`_ function
    * - Butterworth
      - Butterworth filter, using SciPy's `scipy.signal.butter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_ function
    * - Chebyshev I
      - Chebyshev type I filter, using SciPy's `scipy.signal.cheby1 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html>`_ function
    * - Chebyshev II
      - Chebyshev type II filter, using SciPy's `scipy.signal.cheby2 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby2.html>`_ function
    * - Elliptic
      - Elliptic filter, using SciPy's `scipy.signal.ellip <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ellip.html>`_ function

Interpolation
^^^^^^^^^^^^^

Create a new signal which is the interpolation of each selected signal
with respect to a second signal X-axis (which might be the same as one of
the selected signals).

The following interpolation methods are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Method
      - Description
    * - Linear
      - Linear interpolation, using using NumPy's `interp <https://docs.scipy.org/doc/numpy/reference/generated/numpy.interp.html>`_ function
    * - Spline
      - Cubic spline interpolation, using using SciPy's `scipy.interpolate.splev <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splev.html>`_ function
    * - Quadratic
      - Quadratic interpolation, using using NumPy's `polyval <https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyval.html>`_ function
    * - Cubic
      - Cubic interpolation, using using SciPy's `Akima1DInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_ class
    * - Barycentric
      - Barycentric interpolation, using using SciPy's `BarycentricInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.html>`_ class
    * - PCHIP
      - Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation, using using SciPy's `PchipInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_ class

Resampling
^^^^^^^^^^

Create a new signal which is the resampling of each selected signal.

The following parameters are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Parameter
      - Description
    * - Method
      - Interpolation method (see previous section)
    * - Fill value
      - Interpolation fill value (see previous section)
    * - Xmin
      - Minimum X value
    * - Xmax
      - Maximum X value
    * - Mode
      - Resampling mode: step size or number of points
    * - Step size
      - Resampling step size
    * - Number of points
      - Resampling number of points

Detrending
^^^^^^^^^^

Create a new signal which is the detrending of each selected signal.
This features is based on SciPy's `scipy.signal.detrend <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html>`_ function.

The following parameters are available:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Parameter
      - Description
    * - Method
      - Detrending method: 'linear' or 'constant'. See SciPy's `scipy.signal.detrend <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html>`_ function.

Fitting
^^^^^^^

Open an interactive curve fitting tool in a modal dialog box.

.. list-table::
    :header-rows: 1
    :widths: 20, 80

    * - Model
      - Equation
    * - Linear
      - :math:`y = c_{0}+c_{1}.x`
    * - Polynomial
      - :math:`y = c_{0}+c_{1}.x+c_{2}.x^2+...+c_{n}.x^n`
    * - Gaussian
      - :math:`y = y_{0}+\dfrac{A}{\sqrt{2\pi}.\sigma}.exp(-\dfrac{1}{2}.(\dfrac{x-x_{0}}{\sigma})^2)`
    * - Lorentzian
      - :math:`y = y_{0}+\dfrac{A}{\sigma.\pi}.\dfrac{1}{1+(\dfrac{x-x_{0}}{\sigma})^2}`
    * - Voigt
      - :math:`y = y_{0}+A.\dfrac{Re(exp(-z^2).erfc(-j.z))}{\sqrt{2\pi}.\sigma}` with :math:`z = \dfrac{x-x_{0}-j.\sigma}{\sqrt{2}.\sigma}`
    * - Multi-Gaussian
      - :math:`y = y_{0}+\sum_{i=0}^{K}\dfrac{A_{i}}{\sqrt{2\pi}.\sigma_{i}}.exp(-\dfrac{1}{2}.(\dfrac{x-x_{0,i}}{\sigma_{i}})^2)`
    * - Exponential
      - :math:`y = y_{0}+A.exp(B.x)`
    * - Sinusoidal
      - :math:`y = y_{0}+A.sin(2\pi.f.x+\phi)`
    * - Cumulative Distribution Function (CDF)
      - :math:`y = y_{0}+A.erf(\dfrac{x-x_{0}}{\sigma.\sqrt{2}})`

"Computing" menu
----------------

.. figure:: /images/shots/s_computing.png

    Screenshot of the "Computing" menu.

The "Computing" menu allows you to perform various computations on the
selected signals, such as statistics, full width at half-maximum, or
full width at 1/e².

.. note::

    In DataLab vocabulary, a "computing" is a feature that computes a scalar
    result from a signal. This result is stored as metadata, and thus attached
    to signal. This is different from a "processing" which creates a new signal
    from an existing one.

Edit regions of interest
^^^^^^^^^^^^^^^^^^^^^^^^

Open a dialog box to setup multiple Region Of Interests (ROI).
ROI are stored as metadata, and thus attached to signal.

ROI definition dialog is exactly the same as ROI extraction (see above):
the ROI is defined by moving the position and adjusting the width of an
horizontal range.

.. figure:: /images/shots/s_roi_signal.png

    A signal with an ROI.

Remove regions of interest
^^^^^^^^^^^^^^^^^^^^^^^^^^

Remove all defined ROI for selected object(s).

Statistics
^^^^^^^^^^

Compute statistics on selected signal and show a summary table.

.. figure:: /images/shots/s_stats.png

    Example of statistical summary table: each row is associated to an ROI
    (the first row gives the statistics for the whole data).

Histogram
^^^^^^^^^

Compute histogram of selected signal and show it.

Parameters are:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Parameter
      - Description
    * - Bins
      - Number of bins
    * - Lower limit
      - Lower limit of the histogram
    * - Upper limit
      - Upper limit of the histogram

.. figure:: /images/shots/s_histogram.png

    Example of histogram.

Full width at half-maximum
^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute the Full Width at Half-Maximum (FWHM) of selected signal, using one of the following methods:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Method
      - Description
    * - Zero-crossing
      - Find the zero-crossings of the signal after having centered its amplitude around zero
    * - Gauss
      - Fit data to a Gaussian model using least-square method
    * - Lorentz
      - Fit data to a Lorentzian model using least-square method
    * - Voigt
      - Fit data to a Voigt model using least-square method

.. figure:: /images/shots/s_fwhm.png

    The computed result is displayed as an annotated segment.

Full width at 1/e²
^^^^^^^^^^^^^^^^^^

Fit data to a Gaussian model using least-square method.
Then, compute the full width at 1/e².

.. note:: Computed scalar results are systematically stored as metadata.
    Metadata is attached to signal and serialized with it when exporting
    current session in a HDF5 file.

Bandwidth at -3 dB
^^^^^^^^^^^^^^^^^^

Assuming the signal is a filter response, compute the bandwidth at -3 dB by finding the
frequency range where the signal is above -3 dB.

Dynamic parameters
^^^^^^^^^^^^^^^^^^

Compute the following dynamic parameters on selected signal:

.. list-table::
    :header-rows: 1
    :widths: 25, 75

    * - Parameter
      - Description
    * - f
      - Frequency (sinusoidal fit)
    * - ENOB
      - Effective Number Of Bits
    * - SNR
      - Signal-to-Noise Ratio
    * - SINAD
      - Signal-to-Noise And Distortion Ratio
    * - THD
      - Total Harmonic Distortion
    * - SFDR
      - Spurious-Free Dynamic Range

Show results
^^^^^^^^^^^^

Show the results of all computations performed on the selected signals. This shows the
same table as the one shown after having performed a computation.

Plot results
^^^^^^^^^^^^

Plot the results of computations performed on the selected signals, with user-defined
X and Y axes (e.g. plot the FWHM as a function of the signal index).

"View" menu
-----------

.. figure:: /images/shots/s_view.png

    Screenshot of the "View" menu.

The "View" menu allows you to visualize the current signal or group of signals.
It also allows you to show/hide titles, to enable/disable anti-aliasing, or to
refresh the visualization.

View in a new window
^^^^^^^^^^^^^^^^^^^^

Open a new window to visualize and the selected signals.

In the separate window, you may visualize your data more comfortably
(e.g., by maximizing the window) and you may also annotate the data.

.. seealso::
    See :ref:`ref-to-signal-annotations` for more details on annotations.

Show graphical object titles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Show/hide titles of computing results or annotations.

Auto-refresh
^^^^^^^^^^^^

Automatically refresh the visualization when the data changes.
When enabled (default), the plot view is automatically refreshed when the
data changes. When disabled, the plot view is not refreshed until you
manually refresh it by clicking the "Refresh manually" button in the
toolbar. Even though the refresh algorithm is optimized, it may still
take some time to refresh the plot view when the data changes, especially
when the data set is large. Therefore, you may want to disable the
auto-refresh feature when you are working with large data sets,
and enable it again when you are done. This will avoid unnecessary
refreshes.

Refresh manually
^^^^^^^^^^^^^^^^

Refresh the visualization manually. This triggers a refresh of the plot
view, even if the auto-refresh feature is disabled.

Curve anti-aliasing
^^^^^^^^^^^^^^^^^^^

Enable/disable anti-aliasing of curves. Anti-aliasing makes the curves
look smoother, but it may also make them look less sharp.

.. note::
    Anti-aliasing is enabled by default.

.. warning::
    Anti-aliasing may slow down the visualization, especially when
    working with large data sets.

Reset curve styles
^^^^^^^^^^^^^^^^^^

When plotting curves, DataLab automatically assigns a color and a line style to
each curve. Both parameters are chosen from a predefined list of colors and
line styles, and are assigned in a round-robin fashion.

This menu entry allows you to reset the curve styles, so that the next time
you plot curves, the first curve will be assigned the first color and the first
line style of the predefined lists, and the loop will start again from there.

Other menu entries
^^^^^^^^^^^^^^^^^^

Show/hide panels or toolbars.

"?" menu
--------

.. figure:: /images/shots/s_help.png

    Screenshot of the "?" menu.

The "?" menu allows you to access the online documentation, to show log files,
to show information regarding your DataLab installation, and to show the
"About DataLab" dialog box.

Online or Local documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open the online or local documentation:

.. image:: /images/shots/doc_online.png

Show log files
^^^^^^^^^^^^^^

Open DataLab log viewer

.. seealso::
    See :ref:`ref-to-logviewer` for more details on log viewer.

About DataLab installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Show information regarding your DataLab installation (this
is typically needed for submitting a bug report).

.. seealso::
    See :ref:`ref-to-instviewer` for more details on this dialog box.

About
^^^^^

Open the "About DataLab" dialog box:

.. image:: /images/shots/about.png
