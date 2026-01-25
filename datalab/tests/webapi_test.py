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
        from sigima import SignalObj

        from datalab.webapi.serialization import (
            deserialize_object_from_npz,
            serialize_object_to_npz,
        )

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
        from sigima import SignalObj

        from datalab.webapi.serialization import (
            deserialize_object_from_npz,
            serialize_object_to_npz,
        )

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
        from sigima import ImageObj

        from datalab.webapi.serialization import (
            deserialize_object_from_npz,
            serialize_object_to_npz,
        )

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
        from sigima import ImageObj

        from datalab.webapi.serialization import (
            deserialize_object_from_npz,
            serialize_object_to_npz,
        )

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
        from sigima import SignalObj

        from datalab.webapi.serialization import object_to_metadata

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
        from sigima import ImageObj

        from datalab.webapi.serialization import object_to_metadata

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
        from datalab.webapi.schema import ObjectMetadata, ObjectType

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
        from datalab.webapi.schema import ObjectListResponse, ObjectMetadata, ObjectType

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
        from datalab.webapi.schema import MetadataPatchRequest

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
        from datalab.webapi.routes import generate_auth_token

        token1 = generate_auth_token()
        token2 = generate_auth_token()

        # Tokens should be non-empty strings
        assert isinstance(token1, str)
        assert len(token1) > 20

        # Each token should be unique
        assert token1 != token2


# Integration tests would require a running DataLab instance
# They are marked for opt-in execution


@pytest.mark.integration
class TestAPIEndpointsWithMock:
    """Integration tests using a mock workspace adapter."""

    # These tests would use httpx.MockTransport or similar
    # to test the full HTTP flow without a real DataLab instance

    def test_status_endpoint(self):
        """Test the /api/v1/status endpoint (no auth required).

        TODO: Implement with mock adapter - This would set up a test client
        and verify status response.
        """

    def test_list_objects_requires_auth(self):
        """Test that /api/v1/objects requires authentication.

        TODO: Implement with mock adapter - This would verify 401 response
        without token.
        """
