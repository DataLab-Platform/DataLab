.. _settings:

Settings
========

.. meta::
    :description: Settings of DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, settings, scientific, data, analysis, visualization, platform

DataLab provides a comprehensive settings dialog to customize the application behavior,
visualization defaults, and I/O operations. The settings are organized into five tabs:
General, Processing, Visualization, I/O, and Console.

General
-------

The General settings tab contains main window and general feature settings:

.. figure:: /images/shots/settings_main.png

    General settings tab

**Color mode**
    Choose the color mode for the application interface (e.g., light, dark, or auto).

**Process isolation**
    When enabled, each computation runs in a separate process, preventing the application
    from freezing during long computations. This is the recommended setting for better
    responsiveness.

**RPC server**
    Enable the RPC (Remote Procedure Call) server to communicate with external applications,
    such as your own scripts running in Spyder, Jupyter, or other software. This allows
    programmatic control of DataLab.

**Available memory threshold**
    Set a threshold (in MB) below which a warning is displayed before loading new data.
    This helps prevent out-of-memory errors when working with large datasets. Set to 0
    to disable the warning.

**Third-party plugins**
    Enable or disable third-party plugins at startup.

**Plugins path**
    Specify the directory path where DataLab should look for third-party plugins.
    DataLab will also discover plugins in your PYTHONPATH.

Processing
----------

The Processing settings tab controls computation behavior and default parameters:

.. figure:: /images/shots/settings_proc.png

    Processing settings tab

**Operation mode**
    Choose the operation mode for computations taking N inputs:

    - **Single**: single operand mode
    - **Pairwise**: pairwise operation mode

    .. note::

        These operation modes determine how DataLab handles computations involving
        multiple objects. They apply to two types of operations:

        - **N→1 operations**: Combine N (≥2) objects into 1 output (e.g. sum, average)
        - **N+1→N operations**: Apply an operation between N (≥1) objects and 1 operand
          to produce N outputs (e.g. difference, division)

        **Single operand mode** (default): Operations are applied independently within
        each group.

        - For **N→1 operations**: All objects in each group are combined into one result
          per group. Example with groups G1={A, B} and G2={C, D}, sum operation:

          - Result: Σ(A,B) and Σ(C,D) (one per group)

        - For **N+1→N operations**: Each selected object is combined with a single
          reference operand. Example with groups G1={A, B} and G2={C, D}, difference
          with reference R:

          - In G1: A-R, B-R
          - In G2: C-R, D-R

        **Pairwise operation mode**: Objects from different groups are combined at
        matching positions (all groups must have the same number of objects).

        - For **N→1 operations**: Objects at the same position in each group are
          combined. Example with groups G1={A, B} and G2={C, D}, sum operation:

          - Result: A+C and B+D (pairing by position)

        - For **N+1→N operations**: Objects at matching positions are combined pairwise.
          Example with groups G1={A, B}, G2={C, D}, difference with group G3={E, F}:

          - New group 1: A-E, B-F
          - New group 2: C-E, D-F

**Use signal bounds for new signals**
    When enabled, the xmin and xmax values for new signals are initialized from
    the current signal's bounds. When disabled, default values are used.

**Use image dimensions for new images**
    When enabled, the width and height values for new images are initialized from
    the current image's dimensions. When disabled, default values are used.

**FFT shift**
    Enable FFT shift to center the zero-frequency component in the frequency spectrum
    for easier visualization and analysis.

**Extract ROI in single object**
    When enabled, multiple ROIs (Regions of Interest) are extracted into a single object.
    When disabled, each ROI is extracted into a separate object.

**Keep results after computation**
    When enabled, results from previous analyses are kept in the object's metadata after
    computation. When disabled, results are removed. This option is disabled by default
    to avoid confusion from outdated results.

**Ignore warnings**
    Suppress warning messages during computations.

**X-array compatibility behavior**
    Choose the behavior when X arrays are incompatible in multi-signal computations:

    - **Ask**: display a confirmation dialog (default)
    - **Interpolate**: automatically interpolate signals

Visualization
-------------

The Visualization settings tab controls how data is displayed:

.. figure:: /images/shots/settings_view.png

    Visualization settings tab

Common settings
^^^^^^^^^^^^^^^

**Plot toolbar position**
    Choose where to position the plot toolbar (top, bottom, left, or right of the plot).

**Ignore title insertion message**
    Suppress the information message when inserting an object title as an annotation label.

Signal-specific settings
^^^^^^^^^^^^^^^^^^^^^^^^

**Use auto downsampling**
    Enable automatic downsampling for large signals to improve performance and
    visualization clarity.

**Downsampling max points**
    Maximum number of points to display when downsampling is enabled (default: 10,000).

**Autoscale margin**
    Percentage of margin to add around data when auto-scaling signal plots. A value of
    0.2% adds a small margin for better visualization. Set to 0% for no margin (exact
    data bounds).

**DateTime format (s/min/h)**
    Format string for datetime X-axis labels when using standard time units (s, min, h).
    Uses Python's strftime format codes (e.g., %H:%M:%S for hours:minutes:seconds).

**DateTime format (ms/μs/ns)**
    Format string for datetime X-axis labels when using sub-second time units (ms, μs, ns).
    Uses Python's strftime format codes (e.g., %H:%M:%S.%f for hours:minutes:seconds.microseconds).

Image-specific settings
^^^^^^^^^^^^^^^^^^^^^^^

**Lock aspect ratio to 1:1**
    When enabled, the aspect ratio of images is locked to 1:1. When disabled, the aspect
    ratio is determined by the physical pixel size (default and recommended setting).

**Use reference image LUT range**
    When enabled, images are shown with the same LUT (Look-Up Table) range as the first
    selected image, allowing easier comparison.

**Eliminate outliers**
    Percentage of the highest and lowest values to eliminate from the image histogram.
    Recommended values are below 1%.

**Autoscale margin**
    Percentage of margin to add around data when auto-scaling image plots. A value of
    0.2% adds a small margin for better visualization. Set to 0% for no margin (exact
    data bounds).

**Default image visualization settings**
    Click this button to configure default visualization settings for images (colormap,
    interpolation, contrast, etc.).

I/O
---

The I/O settings tab controls input/output operations:

.. figure:: /images/shots/settings_io.png

    I/O settings tab

**Clear workspace before loading HDF5 file**
    When enabled, the workspace is cleared before loading an HDF5 file.

**Ask before clearing workspace**
    When enabled, a confirmation dialog is displayed before clearing the workspace.

**HDF5 full path in title**
    When enabled, the full path of the HDF5 dataset is used as the title for the
    signal/image object. When disabled, only the dataset name is used.

**HDF5 file name in title**
    When enabled, the HDF5 file name is appended as a suffix to the title of the
    signal/image object.

Console
-------

The Console settings tab configures the internal console for debugging and advanced users:

.. figure:: /images/shots/settings_console.png

    Console settings tab

**Console enabled**
    Enable the internal Python console for debugging and advanced scripting.

**Show console on error**
    When enabled, the console is automatically shown when an error occurs in the
    application. This is useful for debugging as it allows you to see the error
    traceback.

**External editor path**
    Path to an external text editor to use for editing Python code from the console.

**External editor arguments**
    Command-line arguments to pass to the external editor.
