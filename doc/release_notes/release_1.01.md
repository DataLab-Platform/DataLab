# Version 1.1 #

## DataLab Version 1.1.0 (2026-02-02) ##

### ✨ New Features ###

**Web API for HTTP/JSON integration:**

DataLab now provides a modern **HTTP/JSON Web API** as an alternative to the existing XML-RPC interface. This is the recommended integration path for new projects, especially for Jupyter notebooks and web-based tools.

* **Accessible via**: File → Web API → Start Web API Server (or set `DATALAB_WEBAPI_ENABLED=1` environment variable)
* **Key benefits**:
  * JSON for metadata and NPZ for binary data (efficient large array transfer)
  * Bearer token authentication for security
  * REST conventions with OpenAPI documentation
  * WASM/Pyodide compatibility (works in browser environments)
* **Endpoints include**:
  * Object listing, retrieval, creation, and deletion
  * Metadata management and binary data transfer (NPZ format)
  * Computation API for running processing operations remotely
* **Auto-discovery**: DataLab-Kernel automatically finds and connects to a running DataLab instance without manual configuration
* **Security**: Localhost binding by default, with optional localhost token bypass for simplified local development
* **Requires additional dependencies**: Install with `pip install datalab-platform[webapi]`

**PyQt6 compatibility:**

* DataLab now fully supports PyQt6 in addition to PyQt5
* Fixed screen geometry retrieval to use `primaryScreen()` for Qt5-Qt6 compatibility
* CI pipeline now includes PyQt6 testing to ensure ongoing compatibility

**Remote control API enhancements:**

> **Note:** These new features also concern the macro commands API, as macros use the same proxy interface as remote control clients.

* Added `call_method()` method to `RemoteProxy` class for calling any public method on DataLab main window or panels:
  * Enables programmatic access to operations not exposed through dedicated proxy methods (e.g., `remove_object`, `delete_all_objects`, `get_current_panel`)
  * Supports automatic method resolution: checks main window first, then current panel if method not found
  * Optional `panel` parameter allows targeting specific panel ("signal" or "image")
  * Thread-safe execution: GUI operations automatically execute in main thread to prevent freezing
  * Proper exception handling: exceptions raised during method execution (e.g., attempting to call private methods) are captured and propagated to the client as `xmlrpc.client.Fault` objects with the original error message
  * New macro example `test_call_method.py` demonstrates usage: removing objects, getting current panel, and panel-specific operations
  * Expands automation capabilities for advanced macro workflows and external application control

* Added `remove_object()` method to proxy interface (local and remote) for selective object deletion
  * Removes currently selected object from active panel
  * Optional `force` parameter to skip confirmation dialog
  * Complements existing `reset_all()` method which clears the entire workspace (including all macros)

**Dependencies:**

* Updated Sigima dependency to version 1.1.0 which includes new features and bug fixes
* Added optional Web API dependencies: FastAPI, Uvicorn, and Pydantic (install with `pip install datalab-platform[webapi]`)
