![DataLab](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/DataLab-banner.png)

[![license](https://img.shields.io/pypi/l/datalab-platform.svg)](./LICENSE)
[![pypi version](https://img.shields.io/pypi/v/datalab-platform.svg)](https://pypi.org/project/datalab-platform/)
[![PyPI status](https://img.shields.io/pypi/status/datalab-platform.svg)](https://github.com/DataLab-Platform/DataLab)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/datalab-platform.svg)](https://pypi.org/project/datalab-platform/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DataLab-Platform/DataLab/binder-environments?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FDataLab-Platform%252FDataLab%26urlpath%3Ddesktop%252F%26branch%3Dbinder-environments)

DataLab is an **open-source platform for scientific and technical data processing
and visualization** with unique features designed to meet industrial requirements.

[**Try DataLab online**](https://mybinder.org/v2/gh/DataLab-Platform/DataLab/binder-environments?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FDataLab-Platform%252FDataLab%26urlpath%3Ddesktop%252F%26branch%3Dbinder-environments), without installing anything, using Binder:

See [DataLab website](https://datalab-platform.com/) for more details.

> **Note:** This project (DataLab Platform) should not be confused with the [datalab-org](https://datalab-org.io/) project, which is a separate and unrelated initiative focused on materials science databases and computational tools.

ℹ️ Created by [CODRA](https://codra.net/)/[Pierre Raybaut](https://github.com/PierreRaybaut) in 2023, developed and maintained by DataLab Platform Developers.

![DataLab](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/shots/i_blob_detection_flower.png)

🧮 DataLab's processing power comes from the advanced algorithms of the object-oriented signal and image processing library [Sigima](https://github.com/DataLab-Platform/Sigima) 🚀 which is part of the DataLab Platform.

![Sigima](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/Sigima-Power.png)

ℹ️ DataLab is powered by [PlotPyStack](https://github.com/PlotPyStack) 🚀 for curve plotting and fast image visualization.

![PlotPyStack](https://raw.githubusercontent.com/PlotPyStack/.github/main/data/plotpy-stack-powered.png)

ℹ️ DataLab is built on Python and scientific libraries.

![Python](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/Python.png) ![NumPy](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/NumPy.png) ![SciPy](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/SciPy.png) ![scikit-image](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/scikit-image.png) ![OpenCV](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/OpenCV.png) ![PlotPyStack](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/plotpystack.png) ![Sigima](https://raw.githubusercontent.com/DataLab-Platform/DataLab/main/doc/images/logos/Sigima.png)

## Key Features

- **Signal processing** (1D): FFT, filtering, fitting, peak detection, stability analysis, and more
- **Image processing** (2D): filtering, morphology, edge detection, blob detection, and more
- **Extensible plugin system** with hot-reload support
- **Macro system** for Python-based automation
- **Remote control** via XML-RPC for integration with Jupyter, Spyder, or any IDE
- **Web API** (HTTP/JSON) for notebook integration and remote control from any HTTP client
- **HDF5 support** for data import/export
- **Batch processing** with ROI (Region of Interest) support

✨ Add features to DataLab by writing your own [plugin](https://datalab-platform.com/en/features/advanced/plugins.html)
(see [plugin examples](https://github.com/DataLab-Platform/DataLab/tree/main/plugins/examples))
or macro (see [macro examples](https://github.com/DataLab-Platform/DataLab/tree/main/macros/examples))

✨ DataLab may be remotely controlled from a third-party application (such as Jupyter,
Spyder or any IDE):

- Using the integrated [remote control](https://datalab-platform.com/en/features/advanced/remote.html)
feature (this requires to install DataLab as a Python package)

- Using the [Web API](https://datalab-platform.com/en/features/advanced/webapi.html)
(HTTP/JSON server for notebook integration and WASM/Pyodide environments)

- Using the lightweight client integrated in [Sigima](https://github.com/DataLab-Platform/Sigima) (`pip install sigima`)

## Installation

DataLab requires **Python 3.9+**.

From [PyPI](https://pypi.org/project/datalab-platform/):

```bash
pip install datalab-platform
```

From [conda-forge](https://anaconda.org/conda-forge/datalab-platform):

```bash
conda install -c conda-forge datalab-platform
```

See the [installation guide](https://datalab-platform.com/en/intro/installation.html) for
more options (standalone installer, WinPython, offline installation, etc.).

----

## Contributing

Contributions are welcome! See the [contributing guide](https://datalab-platform.com/en/contributing/index.html)
or the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.
