# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Routes
==============

FastAPI route definitions for the DataLab Web API.

This module defines all HTTP endpoints for the API. Routes are organized by
function:

- ``/api/v1/status`` - Server status
- ``/api/v1/objects`` - Workspace object operations
- ``/api/v1/objects/{name}/data`` - Binary data transfer

Security
--------

All routes (except ``/api/v1/status``) require bearer token authentication.
The token is generated when the server starts and must be included in the
``Authorization`` header.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status

from datalab.webapi.adapter import WorkspaceAdapter
from datalab.webapi.schema import (
    ApiStatus,
    ErrorResponse,
    MetadataPatchRequest,
    ObjectListResponse,
    ObjectMetadata,
)
from datalab.webapi.serialization import (
    deserialize_object_from_npz,
    object_to_metadata,
    serialize_object_to_npz,
)

# Router for the API
router = APIRouter(prefix="/api/v1", tags=["workspace"])

# Global references (set by controller at startup)
_adapter: WorkspaceAdapter | None = None
_auth_token: str | None = None
_server_url: str | None = None


def set_adapter(adapter: WorkspaceAdapter) -> None:
    """Set the workspace adapter for route handlers."""
    global _adapter  # noqa: PLW0603
    _adapter = adapter


def set_auth_token(token: str) -> None:
    """Set the authentication token."""
    global _auth_token  # noqa: PLW0603
    _auth_token = token


def set_server_url(url: str) -> None:
    """Set the server URL."""
    global _server_url  # noqa: PLW0603
    _server_url = url


def generate_auth_token() -> str:
    """Generate a secure random authentication token."""
    return secrets.token_urlsafe(32)


def get_adapter() -> WorkspaceAdapter:
    """Dependency: Get the workspace adapter."""
    if _adapter is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workspace adapter not initialized",
        )
    return _adapter


def verify_token(authorization: Annotated[str | None, Header()] = None) -> str:
    """Dependency: Verify the bearer token.

    Args:
        authorization: Authorization header value.

    Returns:
        The validated token.

    Raises:
        HTTPException: If token is missing or invalid.
    """
    if _auth_token is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication not configured",
        )

    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Parse "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(parts[1], _auth_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return parts[1]


# =============================================================================
# Status endpoint (no auth required)
# =============================================================================


@router.get("/status", response_model=ApiStatus, tags=["status"])
async def get_status() -> ApiStatus:
    """Get API server status.

    This endpoint does not require authentication and can be used to check
    if the server is running and to get the API version.
    """
    from datalab import __version__

    return ApiStatus(
        running=True,
        version=__version__,
        api_version="v1",
        url=_server_url,
        workspace_mode="live",
    )


# =============================================================================
# Object listing and metadata (JSON control plane)
# =============================================================================


@router.get(
    "/objects",
    response_model=ObjectListResponse,
    responses={401: {"model": ErrorResponse}},
)
async def list_objects(
    _token: str = Depends(verify_token),
    adapter: WorkspaceAdapter = Depends(get_adapter),
) -> ObjectListResponse:
    """List all objects in the workspace.

    Returns metadata for all signals and images currently in DataLab.
    """
    try:
        objects_list = adapter.list_objects()
        result = []

        for name, _panel in objects_list:
            try:
                obj = adapter.get_object(name)
                meta = object_to_metadata(obj, name)
                result.append(ObjectMetadata(**meta))
            except Exception:  # pylint: disable=broad-exception-caught
                # Skip objects that can't be serialized
                continue

        return ObjectListResponse(objects=result, count=len(result))

    except Exception as e:  # pylint: disable=broad-exception-caught
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.get(
    "/objects/{name}",
    response_model=ObjectMetadata,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_object_metadata(
    name: str,
    _token: str = Depends(verify_token),
    adapter: WorkspaceAdapter = Depends(get_adapter),
) -> ObjectMetadata:
    """Get metadata for a specific object.

    Args:
        name: Object name/title.

    Returns:
        Object metadata (without data arrays).
    """
    try:
        obj = adapter.get_object(name)
        meta = object_to_metadata(obj, name)
        return ObjectMetadata(**meta)

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Object '{name}' not found",
        ) from e
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving metadata for '{name}': {e}",
        ) from e


@router.delete(
    "/objects/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def delete_object(
    name: str,
    _token: str = Depends(verify_token),
    adapter: WorkspaceAdapter = Depends(get_adapter),
) -> None:
    """Delete an object from the workspace.

    Args:
        name: Object name/title.
    """
    try:
        adapter.remove_object(name)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Object '{name}' not found",
        ) from e
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting '{name}': {e}",
        ) from e


@router.patch(
    "/objects/{name}/metadata",
    response_model=ObjectMetadata,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def update_object_metadata(
    name: str,
    patch: MetadataPatchRequest,
    _token: str = Depends(verify_token),
    adapter: WorkspaceAdapter = Depends(get_adapter),
) -> ObjectMetadata:
    """Update object metadata.

    Args:
        name: Object name/title.
        patch: Metadata fields to update.

    Returns:
        Updated object metadata.
    """
    try:
        # Convert patch to dict, excluding None values
        updates = patch.model_dump(exclude_none=True)
        adapter.update_metadata(name, updates)

        # Return updated metadata
        obj = adapter.get_object(name)
        meta = object_to_metadata(obj, name)
        return ObjectMetadata(**meta)

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Object '{name}' not found",
        ) from e


# =============================================================================
# Binary data transfer (NPZ format)
# =============================================================================


@router.get(
    "/objects/{name}/data",
    responses={
        200: {"content": {"application/x-npz": {}}},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_object_data(
    name: str,
    _token: str = Depends(verify_token),
    adapter: WorkspaceAdapter = Depends(get_adapter),
) -> Response:
    """Get object data in NPZ format.

    Returns the object's numerical arrays (x/y for signals, data for images)
    plus metadata in a NumPy NPZ archive.

    Args:
        name: Object name/title.

    Returns:
        Binary NPZ archive.
    """
    try:
        obj = adapter.get_object(name)
        npz_data = serialize_object_to_npz(obj)

        # Build Content-Disposition header with safe filename
        # Use RFC 5987 encoding for non-ASCII characters, or fallback to ASCII
        try:
            # Try ASCII first (will fail for Unicode chars)
            name.encode("ascii")
            content_disposition = f'attachment; filename="{name}.npz"'
        except UnicodeEncodeError:
            # Use RFC 5987 encoding for Unicode filenames
            from urllib.parse import quote

            encoded_name = quote(name, safe="")
            content_disposition = f"attachment; filename*=UTF-8''{encoded_name}.npz"

        return Response(
            content=npz_data,
            media_type="application/x-npz",
            headers={
                "Content-Disposition": content_disposition,
            },
        )

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Object '{name}' not found",
        ) from e
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving object '{name}': {e}",
        ) from e


@router.put(
    "/objects/{name}/data",
    status_code=status.HTTP_201_CREATED,
    response_model=ObjectMetadata,
    responses={
        401: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def put_object_data(
    name: str,
    request: Request,
    overwrite: bool = False,
    _token: str = Depends(verify_token),
    adapter: WorkspaceAdapter = Depends(get_adapter),
) -> ObjectMetadata:
    """Create or replace an object from NPZ data.

    The request body must be a NumPy NPZ archive containing the object data
    (x.npy/y.npy for signals, data.npy for images) and metadata.json.

    Args:
        name: Object name/title.
        request: FastAPI request (body contains NPZ archive bytes).
        overwrite: If True, replace existing object.

    Returns:
        Metadata of the created object.
    """
    try:
        # Read body from request
        body = await request.body()

        # Check if exists
        if adapter.object_exists(name) and not overwrite:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Object '{name}' already exists. Use overwrite=true.",
            )

        # Deserialize
        obj = deserialize_object_from_npz(body)
        obj.title = name

        # Add to workspace
        adapter.add_object(obj, overwrite=overwrite)

        # Return metadata
        meta = object_to_metadata(obj, name)
        return ObjectMetadata(**meta)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid NPZ format: {e}",
        ) from e
