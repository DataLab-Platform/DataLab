.. _sig-menu-computing:

Computing features on Signals
=============================

This section describes the signal computing features available in DataLab.

.. seealso::

    :ref:`sig-menu-operations` for more information on operations that can be performed
    on signals, or :ref:`sig-menu-processing` for information on processing features on
    signals.

.. figure:: /images/shots/s_computing.png

    Screenshot of the "Computing" menu.

When the "Signal Panel" is selected, the menus and toolbars are updated to
provide signal-related actions.

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

X values at min/max
^^^^^^^^^^^^^^^^^^^

Compute the X values at minimum and maximum of selected signal.

Peak detection
^^^^^^^^^^^^^^

Create a new signal from semi-automatic peak detection of each selected signal.

.. figure:: /images/shots/s_peak_detection.png

    Peak detection dialog: threshold is adjustable by moving the
    horizontal marker, peaks are detected automatically (see vertical
    markers with labels indicating peak position)

Sampling rate and period
^^^^^^^^^^^^^^^^^^^^^^^^

Compute the sampling rate and period of selected signal.

.. warning:: This feature assumes that the X values are regularly spaced.

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

Bandwidth at -3 dB
^^^^^^^^^^^^^^^^^^

Assuming the signal is a filter response, compute the bandwidth at -3 dB by finding the
frequency range where the signal is above -3 dB.

.. warning::

  This feature assumes that the signal is a filter response, already expressed in dB.

Contrast
^^^^^^^^

Compute the contrast of selected signal.

The contrast is defined as the ratio of the difference and the sum of the maximum
and minimum values:

.. math::
    \text{Contrast} = \dfrac{\text{max}(y) - \text{min}(y)}{\text{max}(y) + \text{min}(y)}

.. note::

  This feature assumes that the signal is a profile from an image, where the contrast
  is meaningful. This justifies the optical definition of contrast.

Show results
^^^^^^^^^^^^

Show the results of all computations performed on the selected signals. This shows the
same table as the one shown after having performed a computation.

Plot results
^^^^^^^^^^^^

Plot the results of computations performed on the selected signals, with user-defined
X and Y axes (e.g. plot the FWHM as a function of the signal index).
