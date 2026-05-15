# Version 1.3 #

## DataLab Version 1.3.0 ##

### ✨ New Features ###

**Third-party plugin discovery via environment variable:**

* Added support for the `DATALAB_PLUGINS` environment variable, allowing one or more directories to be specified as additional plugin search paths
* Multiple directories can be listed using the OS path separator (`;` on Windows, `:` on Linux/macOS), following the same convention as `PYTHONPATH`
* Listed directories are appended to the existing plugin search paths at startup and are picked up automatically by the plugin discovery mechanism
* Non-existent directories are silently skipped (a warning is recorded in the log file), so a stale environment variable on another machine will not prevent DataLab from starting

### 🛠️ Internal changes ###

**History panel:**

* Reworked how processing actions are recorded so that history entries no longer rely on serialised Python callables — sessions saved to disk are now portable across DataLab versions and immune to function refactorings
* Replay now resolves processing functions by name (via the Sigima feature catalog), making history sessions robust to internal code reorganisations as long as the public function names remain stable
