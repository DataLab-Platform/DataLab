.. _sig-menu-file:

Create, open and save Signals
=============================

This section describes how to create, open and save signals (and workspaces).

.. figure:: /images/shots/s_file.png

    Screenshot of the "File" menu.

When the "Signal Panel" is selected, the menus and toolbars are updated to
provide signal-related actions.

The "File" menu allows you to:

- Create, open, save and close signals (see below).

- Save and restore the current workspace or browse HDF5 files (see :ref:`overview`).

- Edit DataLab preferences (see :ref:`settings`).

New signal
^^^^^^^^^^

Create a new signal from various models:

.. list-table::
    :header-rows: 1
    :widths: 20, 80

    * - Model
      - Equation
    * - Zero
      - :math:`y[i] = 0`
    * - Normal distribution
      - :math:`y[i]` is normally distributed with configurable mean and standard deviation
    * - Poisson distribution
      - :math:`y[i]` is Poisson distributed with configurable mean
    * - Uniform distribution
      - :math:`y[i]` is uniformly distributed between two configurable bounds
    * - Gaussian
      - :math:`y = y_{0}+\dfrac{A}{\sqrt{2\pi}.\sigma}.exp(-\dfrac{1}{2}.(\dfrac{x-x_{0}}{\sigma})^2)`
    * - Lorentzian
      - :math:`y = y_{0}+\dfrac{A}{\sigma.\pi}.\dfrac{1}{1+(\dfrac{x-x_{0}}{\sigma})^2}`
    * - Voigt
      - :math:`y = y_{0}+A.\dfrac{Re(exp(-z^2).erfc(-j.z))}{\sqrt{2\pi}.\sigma}` with :math:`z = \dfrac{x-x_{0}-j.\sigma}{\sqrt{2}.\sigma}`
    * - Blackbody (Planck's law)
      - :math:`y = \dfrac{2 h c^2}{\lambda^5 \left(exp\left(\dfrac{h c}{\lambda k T}\right)-1\right)}`
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
    * - Linear chirp
      - :math:`y = y_{0} + A \sin(\phi_{0} + 2\pi (f_{0} x +  \frac{1}{2} c x^{2}))`
    * - Step
      - :math:`y = y_{0}+A.\left\{\begin{array}{ll}1 & \text{if } x > x_{0} \\ 0 & \text{otherwise}\end{array}\right.`
    * - Exponential
      - :math:`y = y_{0}+A.exp(B.x)`
    * - Logistic
      - :math:`y = y_{0} + \dfrac{A}{1 + exp(-k (x - x_{0}))}`
    * - Pulse
      - :math:`y = y_{0}+A.\left\{\begin{array}{ll}1 & \text{if } x_{0} < x < x_{1} \\ 0 & \text{otherwise}\end{array}\right.`
    * - Step Pulse
      - | :math:`y = \left( \begin{cases} y_0 & \text{if } x < t_0 \\ y_0 + A \cdot \dfrac{x - t_0}{t_r} & \text{if } t_0 \leq x < t_0 + t_r \\ y_0 + A & \text{if } x \geq t_0 + t_r \end{cases} \right) + \mathcal{N}\left(0, \sigma_n\right)`

        | where:

        * :math:`t_0` is the pulse start time,
        * :math:`t_r` is the rise time,
        * :math:`\sigma_n` is the noise amplitude

    * - Square Pulse
      - | :math:`y(x) = \left(\begin{cases} y_0 & \text{if } x < t_0 \\ y_0 + A \cdot \dfrac{x - t_0}{t_r} & \text{if } t_0 \leq x < t_0 + t_r \\ y_0 + A & \text{if } t_0 + t_r \leq x < t_1 \\ y_0 + A - A \cdot \dfrac{x - t_1}{t_f} & \text{if } t_1 \leq x < t_1 + t_f \\ y_0 & \text{if } x \geq t_1 + t_f \end{cases} \right) + \mathcal{N}(0, \sigma_n)`

        | where:

        * :math:`t_0` is the pulse start time,
        * :math:`t_r` is the rise time,
        * :math:`t_f` is the fall time,
        * :math:`t_1 = t_0 + t_r + d` is the time at which the decay starts (with :math:`d` the duration of the plateau defined below),
        * :math:`\sigma_n` is the noise amplitude.

        | The duration of the plateau :math:`d` is computed as :math:`d = t_{\mathrm{FWHM}} - \dfrac{t_r + t_f}{2}` from the full width at half maximum :math:`t_{\mathrm{FWHM}}`.

        .. warning::

            The duration of the plateau :math:`d` should not be negative.

    * - Polynomial
      - :math:`y = y_{0}+A_{0}+A_{1}.x+A_{2}.x^2+...+A_{n}.x^n`
    * - Custom
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
    * - FT-Lab files
      - .sig

Open from directory
^^^^^^^^^^^^^^^^^^^

Open multiple signals from a specified directory.

Save signal
^^^^^^^^^^^

Save current signal to the following supported filetypes:

.. list-table::
    :header-rows: 1

    * - File type
      - Extensions
    * - Text files
      - .csv

Save signals to directory
^^^^^^^^^^^^^^^^^^^^^^^^^

Save all selected signals to a specified directory, with configurable filename pattern
and signal format.

.. figure:: /images/shots/s_save_to_directory.png

    Save signals to directory dialog.

When you select "Save to directory..." from the File menu, a dialog appears where you can:

- **Directory**: Choose the target directory where signals will be saved
- **Filename pattern**: Define a pattern for the filenames using Python format strings
- **File extension**: Select the output format (.csv, .txt, etc.)
- **Overwrite**: Choose whether to overwrite existing files
- **Preview**: See the list of files that will be created (with object IDs)

The filename pattern supports the following placeholders:

- ``{title}``: Signal title
- ``{index}``: 1-based index of the signal in the selection (with zero-padding)
- ``{count}``: Total number of selected signals
- ``{xlabel}``, ``{xunit}``, ``{ylabel}``, ``{yunit}``: Axis labels and units
- ``{metadata[key]}``: Access metadata values

You can also use format modifiers, for example ``{index:03d}`` will format the index
with 3 digits zero-padding (001, 002, 003, etc.).

Import text file
^^^^^^^^^^^^^^^^

DataLab can natively import signal files (e.g. CSV, NPY, etc.). However some specific
text file formats may not be supported. In this case, you can use the `Import text file`
feature, which allows you to import a text file and convert it to a signal.

This feature is accessible from the `File` menu, under the `Import text file` option.

It opens an import wizard that guides you through the process of importing the text
file.

Step 1: Select the source
-------------------------

The first step is to select the source of the text file. You can either select a file
from your computer or the clipboard if you have copied the text from another
application.

.. figure:: ../../images/import_text_file/s_01.png
   :alt: Step 1: Select the source
   :align: center

   Step 1: Select the source

Step 2: Preview and configure the import
-----------------------------------------

The second step consists of configuring the import and previewing the result. You can
configure the following options:

- **Delimiter**: The character used to separate the values in the text file.
- **Comments**: The character used to indicate that the line is a comment and should be
  ignored.
- **Rows to Skip**: The number of rows to skip at the beginning of the file.
- **Maximum Number of Rows**: The maximum number of rows to import. If the file contains
  more rows, they will be ignored.
- **Transpose**: If checked, the rows and columns will be transposed.
- **Data type**: The destination data type of the imported data.
- **First Column is X**: If checked, the first column will be used as the X axis.

When you are done configuring the import, click the `Apply` button to see the result.

.. figure:: ../../images/import_text_file/s_02.png
   :alt: Step 2: Configure the import
   :align: center

   Step 2: Configure the import

.. figure:: ../../images/import_text_file/s_03.png
   :alt: Step 2: Preview the result
   :align: center

   Step 2: Preview the result

Step 3: Show graphical representation
-------------------------------------

The third step shows a graphical representation of the imported data. You can use the
`Finish` button to import the data into DataLab workspace.

.. figure:: ../../images/import_text_file/s_04.png
   :alt: Step 3: Show graphical representation
   :align: center

   Step 3: Show graphical representation