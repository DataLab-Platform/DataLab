# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Tests
=============

Unit and integration tests for the DataLab Web API.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sigima import ImageObj, SignalObj

from datalab.webapi.routes import (
    generate_auth_token,
    router,
    set_adapter,
    set_auth_token,
    set_server_url,
)
from datalab.webapi.schema import (
    MetadataPatchRequest,
    ObjectListResponse,
    ObjectMetadata,
    ObjectType,
)
from datalab.webapi.serialization import (
    deserialize_object_from_npz,
    object_to_metadata,
    serialize_object_to_npz,
)

# Check if webapi dependencies are available
try:
    import uvicorn

    WEBAPI_AVAILABLE = uvicorn is not None  # Actually use the import
except ImportError:
    WEBAPI_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not WEBAPI_AVAILABLE, reason="Web API dependencies not installed"
)


class TestNPZSerialization:
    """Tests for NPZ serialization module."""

    def test_signal_round_trip(self):
        """Test serializing and deserializing a SignalObj."""
        # Create a signal
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        obj = SignalObj()
        obj.set_xydata(x, y)
        obj.title = "Test Signal"
        obj.xlabel = "Time"
        obj.ylabel = "Amplitude"
        obj.xunit = "s"
        obj.yunit = "V"

        # Serialize
        data = serialize_object_to_npz(obj)
        assert isinstance(data, bytes)
        assert len(data) > 0

        # Verify it's a valid zip
        buffer = io.BytesIO(data)
        with zipfile.ZipFile(buffer, "r") as zf:
            assert "x.npy" in zf.namelist()
            assert "y.npy" in zf.namelist()
            assert "metadata.json" in zf.namelist()

        # Deserialize
        result = deserialize_object_from_npz(data)

        # Verify
        assert type(result).__name__ == "SignalObj"
        np.testing.assert_array_equal(result.x, x)
        np.testing.assert_array_equal(result.y, y)
        assert result.title == "Test Signal"
        assert result.xlabel == "Time"
        assert result.ylabel == "Amplitude"
        assert result.xunit == "s"
        assert result.yunit == "V"

    def test_signal_with_uncertainties(self):
        """Test signal with dx/dy uncertainties."""
        x = np.linspace(0, 10, 50)
        y = np.cos(x)
        dx = np.ones_like(x) * 0.01
        dy = np.abs(y) * 0.05

        obj = SignalObj()
        obj.set_xydata(x, y, dx=dx, dy=dy)
        obj.title = "Signal with Errors"

        data = serialize_object_to_npz(obj)
        result = deserialize_object_from_npz(data)

        np.testing.assert_array_equal(result.dx, dx)
        np.testing.assert_array_equal(result.dy, dy)

    def test_image_round_trip(self):
        """Test serializing and deserializing an ImageObj."""
        # Create an image
        data = np.random.rand(128, 128).astype(np.float32)
        obj = ImageObj()
        obj.data = data
        obj.title = "Test Image"
        obj.xlabel = "X"
        obj.ylabel = "Y"
        obj.zlabel = "Intensity"
        obj.x0 = 10.0
        obj.y0 = 20.0
        obj.dx = 0.5
        obj.dy = 0.5

        # Serialize
        npz_data = serialize_object_to_npz(obj)
        assert isinstance(npz_data, bytes)

        # Verify structure
        buffer = io.BytesIO(npz_data)
        with zipfile.ZipFile(buffer, "r") as zf:
            assert "data.npy" in zf.namelist()
            assert "metadata.json" in zf.namelist()

        # Deserialize
        result = deserialize_object_from_npz(npz_data)

        # Verify
        assert type(result).__name__ == "ImageObj"
        np.testing.assert_array_equal(result.data, data)
        assert result.title == "Test Image"
        assert result.x0 == 10.0
        assert result.y0 == 20.0
        assert result.dx == 0.5
        assert result.dy == 0.5

    def test_image_preserves_dtype(self):
        """Test that image dtype is preserved through serialization."""
        for dtype in [np.uint8, np.uint16, np.float32, np.float64]:
            obj = ImageObj()
            obj.data = np.random.randint(0, 255, (64, 64)).astype(dtype)
            obj.title = f"Image {dtype.__name__}"

            data = serialize_object_to_npz(obj)
            result = deserialize_object_from_npz(data)

            assert result.data.dtype == dtype, (
                f"Expected {dtype}, got {result.data.dtype}"
            )


class TestObjectMetadata:
    """Tests for object_to_metadata helper."""

    def test_signal_metadata(self):
        """Test extracting metadata from a SignalObj."""
        obj = SignalObj()
        obj.set_xydata(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
        obj.title = "My Signal"
        obj.xlabel = "Time"

        meta = object_to_metadata(obj, "my_signal")

        assert meta["name"] == "my_signal"
        assert meta["type"] == "signal"
        assert meta["shape"] == [100]
        assert meta["dtype"] == "float64"
        assert meta["title"] == "My Signal"
        assert meta["xlabel"] == "Time"

    def test_image_metadata(self):
        """Test extracting metadata from an ImageObj."""
        obj = ImageObj()
        obj.data = np.zeros((256, 512), dtype=np.uint16)
        obj.title = "My Image"
        obj.x0 = 5.0
        obj.dx = 0.1

        meta = object_to_metadata(obj, "my_image")

        assert meta["name"] == "my_image"
        assert meta["type"] == "image"
        assert meta["shape"] == [256, 512]
        assert meta["dtype"] == "uint16"
        assert meta["title"] == "My Image"
        assert meta["attributes"]["x0"] == 5.0
        assert meta["attributes"]["dx"] == 0.1


class TestSchemaModels:
    """Tests for Pydantic schema models."""

    def test_object_metadata_validation(self):
        """Test ObjectMetadata model validation."""
        # Valid metadata
        meta = ObjectMetadata(
            name="test",
            type=ObjectType.SIGNAL,
            shape=[100],
            dtype="float64",
        )
        assert meta.name == "test"
        assert meta.type == ObjectType.SIGNAL

        # With optional fields
        meta_full = ObjectMetadata(
            name="test2",
            type=ObjectType.IMAGE,
            shape=[256, 256],
            dtype="uint8",
            title="Test Image",
            xlabel="X",
            ylabel="Y",
            attributes={"custom": "value"},
        )
        assert meta_full.attributes == {"custom": "value"}

    def test_object_list_response(self):
        """Test ObjectListResponse model."""
        objs = [
            ObjectMetadata(
                name="s1", type=ObjectType.SIGNAL, shape=[50], dtype="float64"
            ),
            ObjectMetadata(
                name="i1", type=ObjectType.IMAGE, shape=[64, 64], dtype="uint8"
            ),
        ]
        response = ObjectListResponse(objects=objs, count=2)

        assert len(response.objects) == 2
        assert response.count == 2

    def test_metadata_patch_request(self):
        """Test MetadataPatchRequest model."""
        patch = MetadataPatchRequest(title="New Title", xlabel="Updated X")

        # model_dump should exclude None values
        data = patch.model_dump(exclude_none=True)
        assert "title" in data
        assert "xlabel" in data
        assert "ylabel" not in data


class TestAuthToken:
    """Tests for authentication token handling."""

    def test_generate_token(self):
        """Test token generation."""
        token1 = generate_auth_token()
        token2 = generate_auth_token()

        # Tokens should be non-empty strings
        assert isinstance(token1, str)
        assert len(token1) > 20

        # Each token should be unique
        assert token1 != token2


# Integration tests would require a running DataLab instance
# They are marked for opt-in execution


class MockWorkspaceAdapter:
    """Mock workspace adapter for testing."""

    def __init__(self):
        self._objects: dict[str, SignalObj | ImageObj] = {}

    def add_object(self, name: str, obj: SignalObj | ImageObj) -> None:
        """Add an object to the mock workspace."""
        self._objects[name] = obj

    def list_objects(self) -> list[tuple[str, str]]:
        """List all objects in the mock workspace."""
        result = []
        for name, obj in self._objects.items():
            panel = "signal" if type(obj).__name__ == "SignalObj" else "image"
            result.append((name, panel))
        return result

    def get_object(self, name: str) -> SignalObj | ImageObj:
        """Get an object by name."""
        if name not in self._objects:
            raise KeyError(f"Object '{name}' not found")
        return self._objects[name]


class TestAPIEndpointsWithMock:
    """Integration tests using a mock workspace adapter."""

    @pytest.fixture
    def test_client(self):
        """Create a test client with mock adapter."""
        # Create a fresh app with the router
        app = FastAPI()
        app.include_router(router)

        # Set up mock adapter and auth
        mock_adapter = MockWorkspaceAdapter()
        test_token = "test-token-12345"

        set_adapter(mock_adapter)
        set_auth_token(test_token)
        set_server_url("http://localhost:8000")

        client = TestClient(app)
        return client, test_token, mock_adapter

    def test_status_endpoint(self, test_client):
        """Test the /api/v1/status endpoint (no auth required)."""
        client, _token, _adapter = test_client

        response = client.get("/api/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert data["running"] is True
        assert "version" in data
        assert data["api_version"] == "v1"
        assert data["url"] == "http://localhost:8000"
        assert data["workspace_mode"] == "live"

    def test_list_objects_requires_auth(self, test_client):
        """Test that /api/v1/objects requires authentication."""
        client, _token, _adapter = test_client

        # Request without Authorization header should return 401
        response = client.get("/api/v1/objects")

        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_list_objects_with_invalid_token(self, test_client):
        """Test that /api/v1/objects rejects invalid tokens."""
        client, _token, _adapter = test_client

        response = client.get(
            "/api/v1/objects", headers={"Authorization": "Bearer wrong-token"}
        )

        assert response.status_code == 401

    def test_list_objects_with_valid_token(self, test_client):
        """Test that /api/v1/objects works with valid token."""
        client, token, adapter = test_client

        # Add a test signal to the mock adapter
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        signal = SignalObj()
        signal.set_xydata(x, y)
        signal.title = "Test Signal"
        adapter.add_object("Test Signal", signal)

        response = client.get(
            "/api/v1/objects", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["objects"]) == 1
        assert data["objects"][0]["name"] == "Test Signal"
        assert data["objects"][0]["type"] == "signal"

    def test_list_objects_empty_workspace(self, test_client):
        """Test listing objects in an empty workspace."""
        client, token, _adapter = test_client

        response = client.get(
            "/api/v1/objects", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["objects"] == []

    def test_invalid_auth_header_format(self, test_client):
        """Test that malformed Authorization headers are rejected."""
        client, _token, _adapter = test_client

        # Missing "Bearer" prefix
        response = client.get(
            "/api/v1/objects", headers={"Authorization": "test-token-12345"}
        )
        assert response.status_code == 401

        # Wrong prefix
        response = client.get(
            "/api/v1/objects", headers={"Authorization": "Basic test-token-12345"}
        )
        assert response.status_code == 401
