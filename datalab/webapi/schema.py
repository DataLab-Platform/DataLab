# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Schema Definitions
==========================

Pydantic models defining the API contract for DataLab Web API.

These schemas define:

- Object metadata (signals and images)
- API request/response payloads
- Event messages

Design Principles
-----------------

1. **Minimal but complete**: Include only essential metadata for workspace operations
2. **Type-safe**: Full type annotations for all fields
3. **Serializable**: All models can be serialized to JSON
4. **Extensible**: Designed for future additions without breaking changes
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ObjectType(str, Enum):
    """Type of data object in the workspace."""

    SIGNAL = "signal"
    IMAGE = "image"


class ObjectMetadata(BaseModel):
    """Metadata for a workspace object.

    This is the core representation of an object in the API.
    The actual data is transferred separately via the binary data plane.
    """

    name: str = Field(..., description="Unique object identifier/title")
    type: ObjectType = Field(..., description="Object type (signal or image)")
    shape: list[int] = Field(..., description="Array shape (e.g., [100] or [512, 512])")
    dtype: str = Field(..., description="NumPy dtype string (e.g., 'float64')")

    # Optional metadata
    title: str | None = Field(None, description="Display title")
    xlabel: str | None = Field(None, description="X-axis label")
    ylabel: str | None = Field(None, description="Y-axis label")
    zlabel: str | None = Field(None, description="Z-axis label (images only)")
    xunit: str | None = Field(None, description="X-axis unit")
    yunit: str | None = Field(None, description="Y-axis unit")
    zunit: str | None = Field(None, description="Z-axis unit (images only)")

    # Extended attributes
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Additional key/value metadata"
    )

    model_config = {"extra": "ignore"}


class ObjectListResponse(BaseModel):
    """Response for listing workspace objects."""

    objects: list[ObjectMetadata] = Field(..., description="List of object metadata")
    count: int = Field(..., description="Total number of objects")


class ObjectCreateRequest(BaseModel):
    """Request to create a new object (metadata only, data sent separately)."""

    name: str = Field(..., description="Object name/title")
    type: ObjectType = Field(..., description="Object type")
    overwrite: bool = Field(False, description="Replace existing object if present")

    # Optional metadata to set
    title: str | None = Field(None, description="Display title")
    xlabel: str | None = Field(None, description="X-axis label")
    ylabel: str | None = Field(None, description="Y-axis label")
    zlabel: str | None = Field(None, description="Z-axis label (images only)")
    xunit: str | None = Field(None, description="X-axis unit")
    yunit: str | None = Field(None, description="Y-axis unit")
    zunit: str | None = Field(None, description="Z-axis unit (images only)")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class MetadataPatchRequest(BaseModel):
    """Request to update object metadata."""

    title: str | None = Field(None, description="New display title")
    xlabel: str | None = Field(None, description="New X-axis label")
    ylabel: str | None = Field(None, description="New Y-axis label")
    zlabel: str | None = Field(None, description="New Z-axis label")
    xunit: str | None = Field(None, description="New X-axis unit")
    yunit: str | None = Field(None, description="New Y-axis unit")
    zunit: str | None = Field(None, description="New Z-axis unit")
    attributes: dict[str, Any] | None = Field(
        None, description="Attributes to merge (not replace)"
    )


class ApiStatus(BaseModel):
    """API server status information."""

    running: bool = Field(..., description="Whether the API server is running")
    version: str = Field(..., description="DataLab version")
    api_version: str = Field("v1", description="API version")
    url: str | None = Field(None, description="Base URL when running")
    workspace_mode: str = Field(..., description="Current workspace mode")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    detail: str | None = Field(None, description="Additional detail")


# Event types for WebSocket (future)
class EventType(str, Enum):
    """Types of workspace events."""

    OBJECT_ADDED = "object_added"
    OBJECT_UPDATED = "object_updated"
    OBJECT_REMOVED = "object_removed"
    OBJECT_RENAMED = "object_renamed"
    WORKSPACE_CLEARED = "workspace_cleared"


class WorkspaceEvent(BaseModel):
    """A workspace change event (for WebSocket notifications)."""

    event: EventType = Field(..., description="Event type")
    object_name: str | None = Field(None, description="Affected object name")
    old_name: str | None = Field(None, description="Old name (for rename events)")
    timestamp: float = Field(..., description="Unix timestamp")
