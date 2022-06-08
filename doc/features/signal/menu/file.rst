"File" menu
===========

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
