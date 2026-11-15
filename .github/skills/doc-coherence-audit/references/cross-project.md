# Cross-Project Checks

Run these checks after the perimeter reports, with the facts they collected. Each one compares statements of several projects; the "Origin" column gives the defect class found by a previous audit (November 2026), to look for again even when that instance was fixed.

| # | Topic | Compare | Origin |
| --- | --- | --- | --- |
| C1 | Same plugin on Desktop and Web | DataLab plugin pages, DataLab-Web `doc/plugins.md` and the Web shim | Promise that one source runs on both hosts, while the shim ignored `register_computations` and the Desktop host never ran `async` callbacks |
| C2 | Operation contracts | Sigima `contracts.rst`, `guidelines.rst`, release notes, Copilot instructions of Sigima and DataLab, and the contract registry (introspected) | Texts stating that contracts "stayed at version 1" while 16 were at version 2 |
| C3 | Remote control scope | DataLab `remote.rst`, Sigima `sigima.client` docstring, SigimaX `remote_control.rst`, the address the server binds | Documentation promising a client on another machine, while the server listens on 127.0.0.1 |
| C4 | Identical results Desktop/Web | DataLab `intro/ecosystem.rst`, DataLab-Web `differences-from-desktop.md`, Sigima contract comparisons, Pyodide NumPy version | "Bit-for-bit" claims contradicted by tolerance comparisons |
| C5 | Object model changes | DataLab menus pages and tutorials, Sigima API pages (e.g. annotations separated from metadata) | Desktop pages describing the previous storage model |
| C6 | Optional dependencies | DataLab and Sigima `requirements.rst`, installation pages, extras of `pyproject.toml`, formats or features silently disabled without an optional package | Extras (`[opencv]`, `[plotly]`) and `pydicom` missing from generated requirement pages |
| C7 | Copied page templates | Installation and contribution pages of DataLab, Sigima, SigimaX | "master branch", stale wheel names, "fork the DataLab repository" in Sigima |
| C8 | Configuration folder | Every mention of `~/.DataLab*` in the four projects | `~/.DataLab/` instead of `~/.DataLab_v<major>/` |
| C9 | Pinned versions | DataLab-Web manifests (`*-dependency.json`), `requirements-dev.txt`, `shim-registry.md`, Desktop `pyproject.toml` files | Documentation citing older minimums than the manifests |
| C10 | Ecosystem pages | DataLab `intro/ecosystem.rst`, `introduction.rst`, `PLATFORM.md`, README files, dependencies of every `pyproject.toml`/`package.json` | A mandatory dependency (DataLab-Capsule) absent from the ecosystem page |
| C11 | Workspace file format | DataLab `h5/`, SigimaX HDF5 pages, DataLab-Web `architecture.md`, `persistence.md`, `differences-from-desktop.md` | "Identical" format claim with undocumented differences (version, group UUID, history dropped by the Web) |
| C12 | Links between sites | DataLab-Web links to the Desktop documentation, Desktop links to the Web and Sigima sites (from the link check) | Links to Desktop pages that do not exist |
| C13 | File formats | DataLab `menu_file.rst` (Signal and Image) and Sigima format lists against the Sigima I/O registries | Listed unsupported formats, missing supported ones, read-only formats not flagged |
| C14 | Dependency classification | Sigima README and Copilot instructions ("no Qt/PlotPyStack dependencies"), DataLab release notes categories | Sigima depends on guidata, classified as PlotPyStack by DataLab |
| C15 | Protocols | SigimaX release notes and pages, DataLab remote control and Web API, DataLab-Kernel | Remote widgets described as HTTP clients of DataLab-Kernel while they use XML-RPC to DataLab |
| C16 | Terminology | Product names in the four projects (`DataLab-Web` vs `DataLab Web`, `DataLab-Kernel`, `DataLab-Capsule`) | Mixed spellings |
| C17 | Generated extracts | Files generated from the code and committed (e.g. DataLab `scripts/datalab_menus.txt`) against their regeneration | Stale menu extract cited by the Copilot instructions |
| C18 | Shared versions | Python, Pyodide, Sigima, guidata, PlotPy, DataLab-Capsule versions in every README, `pyproject.toml`, manifest, classifier, CI matrix and documentation page | Divergent minimum versions between projects |
| C19 | Shared texts | Limits and labels shared by Desktop and Web (capsule limits, provenance verdicts, dialog tab names) | None found in November 2026: keep checking |

Add a row to this table when an audit finds a new class of cross-project defect, and report it in the audit summary.
