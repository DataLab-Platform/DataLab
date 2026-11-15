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
- Requires a bearer token for authentication, except for requests from the local
  machine while the "Web API localhost bypass" setting is enabled (default)
- Token is generated at startup and displayed in the UI

Usage
-----

Enable the Web API via:

- **UI**: File → Web API → Start Web API Server
- **Environment**: ``DATALAB_WEBAPI_ENABLED=1``

See Also
--------

- :mod:`datalab.webapi.controller`: Server lifecycle management
- :mod:`datalab.webapi.routes`: API endpoint definitions
- :mod:`datalab.webapi.adapter`: Thread-safe workspace access
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = [
    "WEBAPI_AVAILABLE",
    "get_webapi_controller",
    "webapi_unavailable_message",
]

try:
    import fastapi  # noqa: F401
    import pydantic  # noqa: F401
    import uvicorn  # noqa: F401
except ImportError as error:
    WEBAPI_AVAILABLE = False
    _WEBAPI_IMPORT_ERROR = error
else:
    WEBAPI_AVAILABLE = True
    _WEBAPI_IMPORT_ERROR = None

_CONTROLLER_INSTANCE = None


def webapi_unavailable_message(_: Callable[[str], str] = str) -> str:
    """Return repair guidance, optionally translated for the GUI.

    Args:
        _: Translation function. Defaults to leaving the message in English.

    Returns:
        The diagnostic in English or translated by the given function.
    """
    return _(
        "Web API is unavailable. FastAPI, Uvicorn and Pydantic are required "
        "dependencies. Check or repair your DataLab installation using the original "
        "installation method or package manager. For pip installations, run "
        "python -m pip check in DataLab's Python environment to check installed "
        "dependency requirements."
    )


def get_webapi_controller():
    """Get the singleton WebAPI controller instance.

    Returns:
        WebApiController instance.

    Raises:
        ImportError: If Web API dependencies or the controller cannot be imported.
    """
    # pylint: disable=global-statement
    global _CONTROLLER_INSTANCE  # noqa: PLW0603

    if not WEBAPI_AVAILABLE:
        raise ImportError(webapi_unavailable_message()) from _WEBAPI_IMPORT_ERROR

    if _CONTROLLER_INSTANCE is None:
        try:
            # pylint: disable=import-outside-toplevel
            from datalab.webapi.controller import WebApiController
        except ImportError as error:
            raise ImportError(webapi_unavailable_message()) from error

        _CONTROLLER_INSTANCE = WebApiController()

    return _CONTROLLER_INSTANCE
