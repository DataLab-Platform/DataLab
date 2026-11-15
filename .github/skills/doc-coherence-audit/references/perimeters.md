# Audit Perimeters

Each perimeter is audited by one subagent (see the prompt template in [report-template.md](./report-template.md)). The lists below are the minimum: the subagent reads every documentation file of its perimeter in full, then confronts each factual statement (menu path, label, default value, signature, command, file name, version, limit, behavior) with the code. "Introspect" means importing the code and reading the real value, not reading the source text only.

Paths are relative to `C:\Dev\Projets\<repo>`. When the tree changed since this list was written (new pages, renamed folders), audit the actual tree and report the list update needed for this file.

## P1. DataLab: Signal and Image Features

- **Documentation**: `doc/features/index.rst`, `doc/features/signal/*.rst`, `doc/features/image/*.rst`, `doc/features/validation/*.rst` and the validation CSV files with their generation script.
- **Code**: `datalab/gui/actionhandler.py`, `datalab/gui/processor/{base,signal,image}.py`, `datalab/gui/panel/base.py`, `datalab/gui/newobject.py`, Sigima `sigima/proc/**`, `sigima/params.py`.
- **Checks**:
  - Regenerate the menus offscreen (`QT_QPA_PLATFORM=offscreen`) with `scripts/print_datalab_menus.py`: every menu entry is documented at the right place, every documented entry exists. The script overwrites `scripts/datalab_menus.txt`: `git diff` it (a diff is itself a finding, the file being cited by the Copilot instructions), then `git restore` it if it was clean before the run.
  - Creation models, parameter names, choices and defaults (introspect the `DataSet` classes).
  - Operation types (1→1, 2→1, n→1, 1→0, 1→n) against the processor registration.
  - Cited algorithms against the Sigima implementation (library function, units, conventions such as degrees or rotation direction).
  - Figures, icons, `literalinclude` targets and `:py:` references exist; `shots/*` images have `.en` and `.fr` variants.
  - Validation CSV links still resolve.

## P2. DataLab: Cross-Cutting Features

- **Documentation**: `doc/features/common/*.rst` (settings, overview, history panel, provenance, capsules, HDF5 browser), `doc/intro/{introduction,keyfeatures,index}.rst`.
- **Code**: `datalab/config/`, `datalab/gui/main.py`, `datalab/gui/panel/history/`, provenance and capsule modules, DataLab-Capsule `src/`, SigimaX HDF5 browser.
- **Checks**:
  - Every option of the Settings dialog: tab, label, real default (introspect the options, not the documentation of the options).
  - Toolbars, actions, icons, shortcuts, context menus of the panels.
  - Limits and constants (sizes, counts, thresholds) against the code of DataLab and DataLab-Capsule.
  - Command line of `workspace-capsule` and public functions (`export_capsule`, `open_capsule`) signatures.
  - Claims of the introduction (feature tables, counts such as "50+ colormaps").

## P3. DataLab: Advanced Topics

- **Documentation**: `doc/features/advanced/*.rst` except the plugin pages (remote control, proxy, Web API, macros, AI assistant, command line, instance viewer, log viewer, model, migration table), `doc/features/advanced/api/*.rst`, `doc/remotecontrol_py27.py`, `macros/examples/*.py`.
- **Code**: `datalab/control/`, `datalab/webapi/`, `datalab/gui/macroeditor*`, `datalab/app.py` (argparse), `datalab/widgets/`.
- **Checks**:
  - Run `datalab --help` (via `run_with_env.py`) and diff its output with `command.rst`.
  - Web API routes: introspect the FastAPI application (method, path, status codes, parameters) against `webapi.rst`; defaults (port, token, localhost bypass, environment variables).
  - Remote API: every method of `BaseProxy` is registered by the XML-RPC server and implemented by `RemoteProxy` and `LocalProxy`, with the documented signature; network scope (bound address) against the documentation.
  - `automodule`/`autoclass` targets import; migration table names exist (introspect each one).
  - Run the macro examples and documented snippets that do not need a GUI; for the others, check each called method exists.

## P4. Plugins (DataLab and DataLab-Web)

- **Documentation**: DataLab `doc/features/advanced/plugins.rst`, `doc/features/advanced/plugins/*.rst`, `doc/intro/tutorials/plugin_catalog.rst`, `datalab/plugins/builtin/_readme_.txt`; DataLab-Web `doc/plugins.md`, `doc/shim-registry.md`, `doc/application-deep-links.md`, `plugins/examples/README.md`; `C:\Dev\Projets\DataLab-Plugins` `README.md`, `CONTRIBUTING.md`, catalog schema.
- **Code**: DataLab `datalab/plugins/`, `datalab/gui/plugins/`, the `datalab-plugin` generator and catalog commands; DataLab-Web `src/runtime/dlplugins/`, `src/runtime/dlw_plugins.py`; DataLab-Plugins `tools/`.
- **Checks**:
  - Discovery order and locations (entry points, user folder `~/.DataLab_v<major>/plugins`, `DATALAB_PLUGINS`, catalogs) against the code.
  - Run the generator into a scratch folder and compare the generated project with the documentation.
  - Run the code examples of the application/recipe pages.
  - Desktop and Web shims: copied modules identical or documented differences; every Desktop plugin API that the Web documentation presents as supported really works in the shim (e.g. `register_computations`, `action_for`, async helpers).
  - Plugin wheel and catalog tooling identical between DataLab and DataLab-Plugins; target Python versions consistent with DataLab classifiers and Pyodide.

## P5. DataLab: Tutorials, Use Cases, Ecosystem

- **Documentation**: `doc/intro/tutorials/**` (except `plugin_catalog.rst`, in P4), `doc/use_cases/**`, `doc/intro/ecosystem.rst`, `doc/index.rst`, `doc/outreach/**`.
- **Code**: menus (regenerated in P1), Sigima parameters, `datalab/data/tutorials`, example plugins, Sigima examples reproducing the tutorials.
- **Checks**:
  - Every menu path and label quoted in a tutorial step exists, under that path.
  - Rerun the key computations of each tutorial on its data and compare with the numbers given in the text and screenshots.
  - Screenshots plausibly current (labels, menus, panels visible in the image): list the outdated ones, do not regenerate them.
  - Ecosystem claims (projects, dependencies between projects, technology stack) against the `pyproject.toml`/`package.json` files.

## P6. DataLab: Installation, Contribution, Meta Files

- **Documentation**: `README.md`, `doc/intro/installation.rst`, `doc/intro/installation_offline.rst`, `doc/requirements.rst`, `requirements.txt`, `doc/contributing/**`, `doc/release_notes/release_<current>.md`, `RELEASING.md`, `CONTRIBUTING.md`, `CITATION.cff`, `.github/copilot-instructions.md`, `.github/instructions/*.md`, `.github/ISSUE_TEMPLATE/*`, `doc/conf.py`.
- **Code**: `pyproject.toml`, `.github/workflows/*.yml`, `.vscode/tasks.json`, `scripts/`, `wix/`, `.env.template`.
- **Checks**:
  - Every installation command: package names on PyPI and conda-forge (query their APIs), extras, branches.
  - Requirements: `pyproject.toml` = `requirements.txt` = `doc/requirements.rst` = README = Copilot instructions "Technology Stack"; the release notes `Requirements` sections match the `pyproject.toml` diff between the release tags.
  - Python versions: classifiers, CI matrix, documentation.
  - Entry points and commands cited exist and accept the cited arguments (`--help`).
  - `RELEASING.md` and the git workflow page against the release workflows (tag pattern, environments, branches).
  - Copilot instructions: tree, file paths, API signatures and rules still true.

## P7. Sigima

- **Documentation**: the whole `doc/` tree (user guide, API, features, contracts, contributing, release notes, examples gallery), `README.md`, `RELEASING.md`, `.github/copilot-instructions.md`.
- **Code**: `sigima/**`, `pyproject.toml`, workflows.
- **Checks**:
  - `features.rst` against `sigima.proc.decorator.find_computation_functions()`: every function exported and documented, no stale entry.
  - Operation contracts: `operation_id` format, versions, qualification (exact vs tolerance), list of v2+ contracts in the release notes against the registry.
  - Run every sphinx-gallery example headless and every documented snippet and docstring example (`doctest`-like run).
  - I/O formats: documented extensions against the format registries (read/write).
  - Contribution pages refer to Sigima (not copied DataLab wording); Python version table current.

## P8. SigimaX

- **Documentation**: the whole `doc/` tree, `README.md`, `RELEASING.md`, `.github/copilot-instructions.md`.
- **Code**: `sigimax/**`, `pyproject.toml`, workflows; usage by DataLab (`DLMainWindow`, options, `sigimax.app.create`).
- **Checks**:
  - Run the gallery examples and the user guide snippets offscreen (unattended mode); check every import of every code block.
  - Lifecycle, configuration, command line options (`--help`, `--version`) against `sigimax/app.py`, `env.py`, `config.py`.
  - HDF5 browser and workspace constants against `sigimax/h5/`.
  - Remote control widgets against the protocol really used (XML-RPC to DataLab).

## P9. DataLab-Web

- **Documentation**: `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `RELEASING.md`, `doc/**/*.md` (including `doc/userguide/` bundled in the application), `packages/sdk/README.md`, `doc/examples/**`.
- **Code**: `src/**` (TypeScript and Python runtime), `packages/sdk/`, `*-dependency.json` manifests, `requirements-dev.txt`, `package.json`, workflows, `scripts/`.
- **Checks**:
  - Every cited component, function, constant, message type, setting and npm script exists, with the cited value.
  - Sigima, guidata and DataLab-Capsule versions: manifests against the documentation and the Desktop requirements; Pyodide and Python versions everywhere.
  - Processing catalog: documented menus against the explicit lists in `src/runtime/processor.py` (`SIGNAL_OVERRIDES`, `IMAGE_OVERRIDES`).
  - SDK events and methods used by the examples (e.g. Angular) are really emitted/exported.
  - User guide links work both on GitHub and in the in-app guide panel.
  - `differences-from-desktop.md` lists every Desktop feature absent or different on the Web (compare with the Desktop menus of P1 and the panels of P2).
