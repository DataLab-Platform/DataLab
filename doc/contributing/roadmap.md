# Roadmap

This document outlines the current and future development plans for DataLab:

* It includes information about [funding sources](#funding), and [future evolution](#future-milestones).
* It also provides a summary of [past milestones](#past-milestones) to give context to the project's evolution.

(funding)=

## 📈 Funding

> ℹ️ As an open-source project, DataLab relies on the support of various organizations and grants to fund its development and maintenance. The project's roadmap is shaped by the needs and priorities of its users, as well as the availability of resources. Funded work is prioritized and scheduled accordingly, while community contributions are welcomed and encouraged.

From the project’s inception in 2023, the development of DataLab has been funded by the following grants and organizations:

| Funding | Description                                                                 |
|---------|-----------------------------------------------------------------------------|
| <a href="https://www.cea.fr/" target="_blank"><img src="../_images/cea.svg" alt="CEA" width="50"/></a> | [CEA](https://www.cea.fr/) - French Alternative Energies and Atomic Energy Commission:<br>• DataLab was initially created for analyzing data from CEA's Laser Megajoule (LMJ) facility<br>• Interfaced with the LMJ control system, DataLab is used to process and visualize signals and images acquired with plasma diagnostics (devices such as cameras, digitizers, spectrometers, etc.)<br>• It is also used for R&D activities around the LMJ facility (e.g., metrology, data analysis, etc.)<br>• CEA is the major investor in DataLab and the main contributor to the project: CEA scientists and engineers are actively involved in the roadmap |
| <a href="https://codra.net/en/offer/software-engineering/datalab/" target="_blank"><img src="../_images/codra.svg" alt="CODRA" width="50"/></a> | [CODRA](https://codra.net/), a software engineering company and software publisher, has supported DataLab’s open-source journey since its inception:<br>• Open-source project management and communication (social media, website, etc.)<br>• Conferences and events: [SciPy 2024](https://cfp.scipy.org/2024/talk/G3MC9L/), [PyData Paris 2024](https://www.youtube.com/watch?v=lBEu-DeHyz0&list=PLJjbbmRgu6RqGMOhahm2iE6NUkIYIaEDK), [Open Source Experience 2024](https://pretalx.com/pydata-paris-2024/talk/WTDVCC/), etc.<br>• Documentation: tutorials, videos, and more |
| <a href="https://nlnet.nl/" target="_blank"><img src="../_images/nlnet.svg" alt="NLnet" width="50"/></a> | [NLnet Foundation](https://nlnet.nl/), as part of the NGI0 Commons Fund backed by the European Commission, funded the [redesign of DataLab’s core architecture](https://nlnet.nl/project/DataLab/) — a major overhaul included in the 1.0 release (November 2025):<br>• The goal was to decouple the data model, computation, and I/O from the UI<br>• This led to the creation of the [Sigima](https://sigima.readthedocs.io/en/latest/) library, which now handles data processing independently of DataLab’s GUI<br>• This redesign improves maintainability, testability, and extensibility of the codebase<br>• This also enables DataLab to be used as a library (through Sigima) in other software projects, fostering integration and reuse |

(future-milestones)=

## 🏗️ Future Milestones

> ℹ️ The following tasks are long-term goals for DataLab. They are not scheduled for any specific release and may evolve over time based on user needs and project direction.

The following table summarizes the future evolutions and maintenance plans for DataLab that are detailed in the sections below.

| Milestone type       | Description                                  |
|----------------------|----------------------------------------------|
| 🔄 Future Evolutions | • Support for data acquisition<br>• Support for time series<br>• Database connectors<br>• Spyder plugin for interactive data analysis |
| 🛠️ Maintenance      | • Transition to gRPC for remote control<br>• Drop Qt5 support and migrate to Qt6 |
| 🧱 Other Tasks       | • Create a DataLab plugin template          |

### 🔄 Support for Data Acquisition

Adding support for data acquisition would be a major step forward in making DataLab not only a platform for signal and image processing, but also a **versatile tool for real-time experimental workflows**. The idea would be to allow users to acquire data directly from various hardware devices (e.g., cameras, digitizers, spectrometers, etc.) **within DataLab itself**.

Such a feature would enable **seamless integration between acquisition and analysis**, allowing users to process and visualize data immediately after it is captured, without needing to switch tools or export/import files.

While no formal design has been established yet, a viable approach could be to:

* Introduce a **new plugin family dedicated to data acquisition**, following the modular architecture of DataLab;
* Define a **generic API for acquisition plugins**, ensuring flexibility and compatibility across device types;
* **Leverage existing solutions** such as [PyMoDAQ](https://pymodaq.cnrs.fr/), a Python-based data acquisition framework already compatible with a wide range of laboratory instruments.

A potential **collaboration with the PyMoDAQ development team** could be explored, to benefit from their ecosystem and avoid duplicating efforts. This would also help foster interoperability and promote open standards in the Python scientific instrumentation community.

### 🌐 Web Frontend ✅ Delivered

A web frontend has been delivered as **DataLab-Web**, the browser-native edition of the platform (see the {ref}`ecosystem overview <ecosystem>`). Rather than running on a server, it runs the [Sigima](https://sigima.readthedocs.io/en/latest/) computation engine **entirely inside the browser** — CPython compiled to WebAssembly via [Pyodide](https://pyodide.org/) — paired with a React user interface. This keeps DataLab a **local application**: no server, no account, and the data never leaves the user's machine, consistent with the project's privacy-first philosophy. Try it at [datalab-platform.com/web](https://datalab-platform.com/web/).

### ⏱️ Support for Time Series

DataLab currently focuses on generic signal and image processing, but many use cases — especially in scientific instrumentation and experimental physics — involve **time series data**.

Adding dedicated support for time series would make it easier to:

* Handle signals with associated timestamps or non-uniform sampling;
* Perform time-aware processing and visualization (e.g., event alignment, time-based filtering, resampling);
* Integrate with external systems generating time-indexed data.

This feature is tracked in [Issue #27](https://github.com/DataLab-Platform/DataLab/issues/27), where potential use cases and design considerations are discussed.

Introducing robust time series handling would broaden DataLab’s applicability in domains such as data logging, slow control, and real-time monitoring.

### 🗃️ Database Connectors

Currently, DataLab operates primarily on files and in-memory data structures. Adding **connectors to databases** would significantly extend its capabilities, especially for users dealing with large, structured, or historical datasets.

This feature would allow DataLab to:

* **Query and load data** from SQL or NoSQL databases (e.g. PostgreSQL, SQLite, MongoDB, etc.);
* Support metadata-driven workflows, where experiments or datasets are referenced from a database;
* **Store analysis results** or annotations back into a database for traceability and reproducibility;
* Facilitate integration into lab information management systems (LIMS) or enterprise data infrastructures.

The design could include:

* A **plugin-based system** for supporting various database backends;
* A simple configuration interface for connection settings and query management;
* Integration with pandas or SQLAlchemy for flexible data exchange.

This evolution would help bridge the gap between **data acquisition, analysis, and long-term storage**, enabling more robust scientific data workflows.

### 📓 Jupyter Plugin for Interactive Data Analysis

Although DataLab can already be remotely controlled from a Jupyter notebook — thanks to its existing remote control capabilities (see {ref}`ref-to-remote-control`) — the user experience could be greatly enhanced by developing a **dedicated Jupyter plugin**.

This plugin would provide a **tighter integration** between Jupyter and DataLab, offering the following features:

* Use DataLab as a **Jupyter kernel**, enabling direct access to its processing capabilities from within a notebook;
* **Display numerical results from DataLab in Jupyter**, and vice versa — for example, importing results computed in a Jupyter notebook into the DataLab interface;
* Allow **interactive data manipulation**: use DataLab for efficient signal and image operations, and Jupyter for custom or home-made data analysis routines;
* Bridge scripting and GUI workflows, making DataLab more attractive for scientific users familiar with Jupyter environments.

Technically, this plugin could rely on the **Jupyter kernel interface** to expose DataLab's capabilities in an interactive programming context.

Such integration would reinforce DataLab’s role in the scientific Python ecosystem and facilitate reproducible, notebook-driven analysis workflows.

### 🧩 Spyder Plugin for Interactive Data Analysis

A plugin for the [Spyder IDE](https://www.spyder-ide.org/) would address the **same use cases as the Jupyter plugin**, providing seamless integration between DataLab and Spyder’s interactive environment.

This plugin would allow users to:

* Interact with DataLab directly from Spyder;
* Visualize or send data between the DataLab GUI and the Spyder console;
* Use DataLab’s processing capabilities in real time while scripting custom analysis workflows in Spyder.

As with the Jupyter integration, this plugin could be implemented by leveraging the **Jupyter kernel interface** (see {ref}`ref-to-remote-control`), which Spyder already supports internally.

Such a plugin would enhance DataLab's usability for scientists and engineers who prefer Spyder’s integrated development environment for exploratory analysis.

### 🧠 Jupyter Kernel Interface for DataLab ✅ Delivered

> ℹ️ A Jupyter kernel has been delivered as **DataLab-Kernel** (see its [documentation](https://datalab-kernel.readthedocs.io/) and the {ref}`ecosystem overview <ecosystem>`), providing notebook-driven access to DataLab workspaces with optional live synchronization to the desktop GUI. The original exploration notes are kept below for context.

Implementing a native **Jupyter kernel interface** for DataLab would provide a more integrated way to use it from other environments such as **Jupyter notebooks, Spyder, or Visual Studio Code**.

This interface would allow:

* Direct control of DataLab from third-party tools that support Jupyter kernels;
* **Two-way data exchange**: e.g., displaying DataLab results inside a notebook, or visualizing Jupyter-generated data inside the DataLab GUI;
* Tighter integration into scripting workflows and IDEs beyond simple remote control.

However, based on initial exploration, implementing a full Jupyter kernel may be **non-trivial and potentially time-consuming**. Given that remote control already enables communication between DataLab and Jupyter (see {ref}`ref-to-remote-control`), the added value of a full kernel integration should be carefully evaluated against its complexity and maintenance cost.

This option remains open for discussion depending on user demand and development resources.

### 🛠️ Maintenance Plan

#### 🔄 2026: Transition to gRPC for Remote Control

DataLab currently relies on XML-RPC for remote control, which may become a limitation for more advanced or high-performance use cases. If the need arises for a more **efficient, robust, and extensible communication protocol**, a switch to **gRPC** is under consideration.

This improvement is tracked in [Issue #18](https://github.com/DataLab-Platform/DataLab/issues/18).

#### 🚫 2025: Drop Qt5 Support and Migrate to Qt6

With the **end-of-life of Qt5 scheduled for mid-2025**, DataLab will fully migrate to **Qt6**.
This transition is expected to be **straightforward**, thanks to:

* The use of the `qtpy` abstraction layer;
* The fact that the `PlotPyStack` library is already compatible with Qt6.

This change will ensure compatibility with future versions of Qt and modern Python environments.

### 🧱 Other Tasks

#### 🧩 Create a DataLab Plugin Template

To encourage community contributions and facilitate the development of extensions, a **template for creating DataLab plugins** is planned.

This template would:

* Provide a ready-to-use scaffold with best practices;
* Help new developers quickly understand the plugin architecture;
* Promote consistency and modularity across third-party plugins.

The task is tracked in [Issue #26](https://github.com/DataLab-Platform/DataLab/issues/26).

(past-milestones)=

## 🏆 Past Milestones

From version 0.9 to 1.0, DataLab has undergone significant development and enhancements. The project has evolved from a simple data analysis tool to a powerful platform for processing and visualizing signals and images.

Those enhancements have been made possible thanks to the support of the following organizations (see [Funding](#funding) for more details):

* [CEA](https://www.cea.fr/): French Alternative Energies and Atomic Energy Commission
* [CODRA](https://codra.net/en/): software engineering company and software publisher
* [NLnet](https://nlnet.nl/): NLnet Foundation, as part of the NGI0 Commons Fund

The following table summarizes the major past milestones of DataLab, including the release dates and a brief description of the features or enhancements introduced in each version.

| Milestone | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| [1.0](https://github.com/DataLab-Platform/DataLab/releases/tag/v1.0.0)<br>2025/11 | • Interactive object editing: modify creation/processing parameters and recompute objects<br>• New object creation: parametric signal/image generators, Poisson noise, complex-valued objects<br>• Enhanced processing: pulse features, ideal filters, 2D resampling, convolution, projections<br>• ROI & annotations: copy/paste, import/export, grid creation, inverse ROI logic<br>• Visualization: merged result labels, datetime support, performance optimizations<br>• File formats: FT-Lab (.sig, .ima), coordinated text files, improved HDF5 support<br>• Non-uniform coordinates and polynomial calibration<br>• Public API enhancements for programmability |
| [0.20](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.20.0)<br>2025/04 | • ANDOR SIF images: add support for background images<br>• Array editor: new copy all, export, and paste features<br>• Fourier analysis: new zero padding feature for signals and images<br>• ROI editor: ROI selection tool now active by default<br>• Signal processing: X-Y mode, abscissa/ordinate find features, full width at y=...<br>• Public API: add `group_id` and `set_current` arguments to add methods |
| [0.19](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.19.0)<br>2025/04 | • Open all signals or images from a folder (recursively)<br>• ROI editor: add options to create ROIs from coordinates |
| [0.18](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.18.0)<br>2024/11 | • New pairwise operand mode (the operation is done on each pair of signals/images)<br>• New polygonal ROI feature<br>• Support Windows 7 SP1 to Windows 11 with a single installer |
| [0.17](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.17.0)<br>2024/08 | • Introduce ROI support across all processing features<br>• Add arithmetic operations on signals and images<br>• New Ubuntu package for native installation on Linux<br>• New Conda package for all platforms (Windows, Linux, MacOS) |
| [0.16](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.16.0)<br>2024/06 | • New validation process for signal and image features<br>• Add support for binary images (unlocking new processing features) |
| [0.15](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.15.0)<br>2024/04 | • New MSI installer for the stand-alone version on Windows<br>• Add support for large text/CSV files (> 1 GB)<br>• Add auto downsampling feature |
| [0.14](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.14.0)<br>2024/03 | • HDF5 browser: multiple file support, detailed info on groups and attributes<br>• New Debian package for native installation on Linux |
| [0.12](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.12.0)<br>2024/02 | • Add a tour and demo feature<br>• Add tutorials to the documentation<br>• Add a Text file import assistant |
| [0.11](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.11.0)<br>2024/01 | • Add features for reordering signals and images (e.g. drag-and-drop)<br>• Add 1D convolution, interpolation, resampling and detrending features |
| [0.10](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.10.0)<br>2023-12 | • Develop a very simple DataLab plugin to demonstrate the plugin system<br>• Allow to disable the automatic refresh when doing multiple processing steps<br>• Serialize curve and image styles in HDF5 files<br>• Improve curve readability |
| [0.9](https://github.com/DataLab-Platform/DataLab/releases/tag/v0.9.0)<br>2023-11 | • Run computations in a separate process<br>• Add a plugin system: API for third-party extensions<br>• Add a macro-command system<br>• Add an XML-RPC server to allow DataLab remote control |
