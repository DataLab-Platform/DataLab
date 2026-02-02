.. _ref-to-webapi:

Web API
=======

.. meta::
    :description: How to use the DataLab Web API for HTTP/JSON integration
    :keywords: DataLab, Web API, REST, HTTP, JSON, Jupyter, notebook, integration

The DataLab Web API provides a modern **HTTP/JSON interface** for external access
to the DataLab workspace. It is the recommended integration path for new projects,
especially for Jupyter notebooks and web-based tools.

.. note::

    The Web API is available starting from DataLab 1.1. The required dependencies
    (FastAPI, Uvicorn, Pydantic) are included in the standard DataLab installation.

Overview
--------

The Web API enables:

- **Jupyter notebook integration** via `DataLab-Kernel <https://github.com/DataLab-Platform/DataLab-Kernel>`_
- **WASM/Pyodide compatibility** (no XML-RPC dependency)
- **External tool integration** (any HTTP client)

Unlike the XML-RPC interface, the Web API:

- Uses **JSON** for metadata and **NPZ** for binary data (efficient large array transfer)
- Provides **bearer token authentication** for security
- Follows **REST conventions** with OpenAPI documentation
- Works in **browser environments** (WASM/Pyodide)

.. list-table:: Comparison with XML-RPC
   :header-rows: 1

   * - Feature
     - XML-RPC
     - Web API
   * - Protocol
     - XML
     - JSON + Binary (NPZ)
   * - WASM Support
     - ❌
     - ✅
   * - Large Arrays
     - Slow (base64)
     - Fast (NPZ)
   * - Authentication
     - None
     - Bearer Token
   * - Standards
     - XML-RPC
     - REST + OpenAPI

Quick Start
-----------

Enabling the Web API
^^^^^^^^^^^^^^^^^^^^

There are several ways to enable the Web API server:

1. **Via UI**: File → Web API → Start Web API Server
2. **Via environment variable**: Set ``DATALAB_WEBAPI_ENABLED=1`` before starting DataLab

When started, DataLab displays the server URL and authentication token in a dialog.
The status bar also shows the Web API port when the server is running.

The default port is **18080** (or the next available port if busy).

.. TODO: Add screenshot when available
   .. figure:: /images/shots/webapi_started.png

       Web API connection dialog showing URL and token

Auto-Discovery
^^^^^^^^^^^^^^

DataLab-Kernel can automatically discover and connect to a running DataLab instance
without any manual configuration. When the Web API starts, DataLab writes connection
information to a file that DataLab-Kernel reads automatically.

**Just load the extension** — no environment variables or explicit connection needed:

.. code-block:: python

    # In your notebook (JupyterLab, VS Code, etc.)
    %load_ext datalab_kernel

    # DataLab-Kernel automatically finds and connects to DataLab
    workspace.list()  # Already connected!

The auto-discovery mechanism tries the following methods in order:

1. **Environment variables** (``DATALAB_WORKSPACE_URL``, ``DATALAB_WORKSPACE_TOKEN``)
2. **Connection file** written by DataLab (native Python only)
3. **URL query parameters** (JupyterLite: ``?datalab_url=...&datalab_token=...``)
4. **Well-known port probing** (``http://127.0.0.1:18080``)

If discovery fails, DataLab-Kernel starts in standalone mode and you can connect
later using ``workspace.connect()``.

Manual Connection (Legacy)
^^^^^^^^^^^^^^^^^^^^^^^^^^

If auto-discovery doesn't work (e.g., running on a different machine), you can
set environment variables before starting your notebook kernel:

.. code-block:: bash

    export DATALAB_WORKSPACE_URL=http://127.0.0.1:18080
    export DATALAB_WORKSPACE_TOKEN=<your-token>

Then in your notebook using DataLab-Kernel:

.. code-block:: python

    from datalab_kernel import workspace

    # List objects in DataLab
    workspace.list()

    # Retrieve an object
    signal = workspace.get("my_signal")

    # Add a processed result back to DataLab
    workspace.add("processed", result)

API Reference
-------------

Base URL: ``http://127.0.0.1:<port>/api/v1``

Status Endpoint (No Auth Required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Method
     - Endpoint
     - Description
   * - GET
     - ``/status``
     - Server status and version information

Object Operations (Auth Required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All endpoints below require the ``Authorization: Bearer <token>`` header.

.. list-table::
   :header-rows: 1

   * - Method
     - Endpoint
     - Description
   * - GET
     - ``/objects``
     - List all objects with metadata (JSON)
   * - GET
     - ``/objects/{name}``
     - Get object metadata (JSON)
   * - DELETE
     - ``/objects/{name}``
     - Delete an object
   * - PATCH
     - ``/objects/{name}/metadata``
     - Update object metadata

Binary Data Transfer (Auth Required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Method
     - Endpoint
     - Description
   * - GET
     - ``/objects/{name}/data``
     - Download object as NPZ archive
   * - PUT
     - ``/objects/{name}/data``
     - Upload object from NPZ archive

Data Format
-----------

Objects are transferred using NumPy's NPZ format for efficiency.

SignalObj NPZ Structure
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    my_signal.npz
    ├── x.npy          # X coordinates (float64)
    ├── y.npy          # Y data (float64)
    ├── dx.npy         # X uncertainties (optional)
    ├── dy.npy         # Y uncertainties (optional)
    └── metadata.json  # Labels, units, title

ImageObj NPZ Structure
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    my_image.npz
    ├── data.npy       # 2D image array (preserves dtype)
    └── metadata.json  # Labels, units, coordinates (x0, y0, dx, dy)

Authentication
--------------

All API endpoints except ``/status`` require a bearer token:

.. code-block:: text

    Authorization: Bearer <token>

The token is generated when the server starts and displayed in the connection dialog.

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
     - Default
   * - ``DATALAB_WEBAPI_HOST``
     - Bind address
     - ``127.0.0.1``
   * - ``DATALAB_WEBAPI_PORT``
     - Server port
     - ``18080`` (or next available)
   * - ``DATALAB_WEBAPI_TOKEN``
     - Auth token
     - Generated

Security Model
--------------

The Web API implements the following security measures:

1. **Localhost binding**: By default, the server only accepts connections from
   the local machine (127.0.0.1).

2. **Token authentication**: Every request (except status) must include a valid
   bearer token.

3. **Explicit opt-in**: Remote binding (0.0.0.0) requires explicit configuration.

Localhost Token Bypass
^^^^^^^^^^^^^^^^^^^^^^

For simplified local development, you can disable token verification for localhost
connections in **Edit → Settings → Web API localhost bypass**.

When enabled, clients connecting from ``127.0.0.1`` do not need to provide a token.
This makes auto-discovery work seamlessly even when the connection file cannot be
read (e.g., in JupyterLite or sandboxed environments).

.. warning::

    Only enable localhost bypass when you control all applications running on
    your machine. Malicious local software could access your data without a token.

.. warning::

    Exposing the API to the network allows anyone with the token to access your
    data. Use with caution on trusted networks only.

Architecture
------------

.. code-block:: text

    ┌─────────────────────────────────────────┐
    │           DataLab Main Window           │
    │                 (Qt)                    │
    ├─────────────────────────────────────────┤
    │            WebApiController             │
    │         (Server Lifecycle)              │
    ├─────────────────────────────────────────┤
    │            WorkspaceAdapter             │
    │         (Thread-safe access)            │
    ├─────────────────────────────────────────┤
    │          FastAPI + Uvicorn              │
    │         (Separate thread)               │
    └─────────────────────────────────────────┘
                      ↑
                      │ HTTP/JSON
                      ↓
    ┌─────────────────────────────────────────┐
    │        DataLab-Kernel / Client          │
    └─────────────────────────────────────────┘

The Web API runs in a separate thread from the Qt GUI. The ``WorkspaceAdapter``
ensures thread-safe access to DataLab objects by marshalling calls to the main
thread using Qt's signal/slot mechanism.

Installation
------------

The Web API dependencies are included in the standard DataLab installation:

- FastAPI (web framework)
- Uvicorn (ASGI server)
- Pydantic (data validation)

No additional installation steps are required.

Programmatic Control
--------------------

The Web API can also be started and controlled programmatically via the XML-RPC
interface, which is useful for automated workflows:

.. code-block:: python

    from datalab.control.remote import RemoteClient

    proxy = RemoteClient()
    proxy.connect()

    # Start the Web API server
    info = proxy.start_webapi_server()
    print(f"URL: {info['url']}")
    print(f"Token: {info['token']}")

    # Check status
    status = proxy.get_webapi_status()
    print(f"Running: {status['running']}")

    # Stop the server
    proxy.stop_webapi_server()

See Also
--------

- :ref:`ref-to-remote-control` for the XML-RPC interface (legacy)
- `DataLab-Kernel documentation <https://github.com/DataLab-Platform/DataLab-Kernel>`_ for Jupyter integration
