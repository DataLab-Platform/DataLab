# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
DataLab Web API
===============

This package provides a web-native HTTP/JSON API for DataLab, enabling:

- **DataLab-Kernel integration**: Jupyter notebooks can connect to a running
  DataLab instance via HTTP instead of XML-RPC
- **WASM/Pyodide compatibility**: The HTTP-based protocol works in WebAssembly
  environments where XML-RPC is not available
- **External tool integration**: Any HTTP client can interact with the DataLab
  workspace

Architecture
------------

The Web API follows a layered design:

- **Control plane (JSON)**: Metadata operations via standard REST endpoints
- **Data plane (binary)**: Efficient NumPy array transfer using NPZ format
- **Events (WebSocket)**: Optional real-time notifications (future)

Security
--------

By default, the API:

- Binds to localhost only (127.0.0.1)
- Requires a bearer token for authentication
- Token is generated at startup and displayed in the UI

Usage
-----

Enable the Web API via:

- **UI**: Tools → Web API → Start
- **CLI**: ``datalab --webapi``
- **Environment**: ``DATALAB_WEBAPI_ENABLED=1``

See Also
--------

- :mod:`datalab.webapi.controller`: Server lifecycle management
- :mod:`datalab.webapi.routes`: API endpoint definitions
- :mod:`datalab.webapi.adapter`: Thread-safe workspace access
"""

from __future__ import annotations

__all__ = [
    "WEBAPI_AVAILABLE",
    "get_webapi_controller",
]

# Check if webapi dependencies are available
try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401

    WEBAPI_AVAILABLE = True
except ImportError:
    WEBAPI_AVAILABLE = False

_CONTROLLER_INSTANCE = None


def get_webapi_controller():
    """Get the singleton WebAPI controller instance.

    Returns:
        WebApiController instance, or None if webapi dependencies not installed.

    Raises:
        ImportError: If webapi dependencies are not available.
    """
    # pylint: disable=global-statement
    global _CONTROLLER_INSTANCE  # noqa: PLW0603

    if not WEBAPI_AVAILABLE:
        raise ImportError(
            "Web API dependencies not installed. "
            "Install with: pip install datalab-platform[webapi]"
        )

    if _CONTROLLER_INSTANCE is None:
        # pylint: disable=import-outside-toplevel
        from datalab.webapi.controller import WebApiController

        _CONTROLLER_INSTANCE = WebApiController()

    return _CONTROLLER_INSTANCE
