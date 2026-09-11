# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Tests - Extra Coverage
==============================

Complementary tests for the DataLab Web API targeting paths not exercised
by ``webapi_test.py``:

- NPZ serialization edge cases (``compress=False``, errors, numpy metadata
  round-trip, ``coords`` field, missing files, unsupported types).
- Routes integration tests with a richer ``MockWorkspaceAdapter`` covering
  authentication branches and all CRUD endpoints (GET/PATCH/DELETE/PUT/POST).
"""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace
from typing import Union
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sigima import ImageObj, SignalObj

from datalab.gui.processor.base import PROCESSING_PARAMETERS_OPTION
from datalab.objectmodel import get_uuid
from datalab.webapi.adapter import WorkspaceAdapter
from datalab.webapi.routes import (
    router,
    set_adapter,
    set_auth_token,
    set_localhost_no_token,
    set_server_url,
)
from datalab.webapi.serialization import (
    _deserialize_obj_metadata,
    _make_json_serializable,
    _to_python_scalar,
    deserialize_object_from_npz,
    object_to_metadata,
    serialize_object_to_npz,
)

# =============================================================================
# Helpers
# =============================================================================

DataObject = Union[SignalObj, ImageObj]


def _make_signal(title: str = "S", n: int = 32) -> SignalObj:
    """Return a small SignalObj for tests."""
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x)
    obj = SignalObj()
    obj.set_xydata(x, y)
    obj.title = title
    return obj


def _make_image(title: str = "I", shape: tuple[int, int] = (16, 24)) -> ImageObj:
    """Return a small ImageObj for tests."""
    obj = ImageObj()
    obj.data = np.arange(shape[0] * shape[1], dtype=np.uint16).reshape(shape)
    obj.title = title
    return obj


class ObjectListModel(list):
    """Minimal object model used to exercise the real workspace adapter."""

    def remove_object(self, obj: DataObject) -> None:
        """Remove an object from the model."""
        self.remove(obj)


def make_panel(objects: list[DataObject]) -> SimpleNamespace:
    """Build the panel surface required by :class:`WorkspaceAdapter`."""
    return SimpleNamespace(
        objmodel=ObjectListModel(objects),
        plothandler=SimpleNamespace(remove_item=Mock()),
        objview=SimpleNamespace(
            remove_item=Mock(), update_tree=Mock(), select_objects=Mock()
        ),
        SIG_OBJECT_REMOVED=SimpleNamespace(emit=Mock()),
        SIG_OBJECT_MODIFIED=SimpleNamespace(emit=Mock()),
        SIG_REFRESH_PLOT=SimpleNamespace(emit=Mock()),
    )


def test_workspace_adapter_resolves_uuid_and_rejects_ambiguous_titles() -> None:
    """Use stable UUIDs for mutations and reject ambiguous title lookups."""
    signal = _make_signal("Duplicate", n=10)
    image = _make_image("Duplicate")
    signal_panel = make_panel([signal])
    image_panel = make_panel([image])
    adapter = WorkspaceAdapter(
        SimpleNamespace(signalpanel=signal_panel, imagepanel=image_panel)
    )
    signal_uuid = get_uuid(signal)
    image_uuid = get_uuid(image)
    processing_parameters = {
        "func_name": "derivative",
        "pattern": "1-to-1",
        "source_uuid": signal_uuid,
    }
    signal.set_metadata_option(PROCESSING_PARAMETERS_OPTION, processing_parameters)

    assert adapter.get_object_panel(signal_uuid) == "signal"
    fetched = adapter.get_object(signal_uuid)
    assert fetched.get_metadata_option(PROCESSING_PARAMETERS_OPTION) == (
        processing_parameters
    )
    with pytest.raises(ValueError, match="ambiguous"):
        adapter.get_object_panel("Duplicate")
    with pytest.raises(ValueError, match="ambiguous"):
        adapter.select_objects(["Duplicate"])

    selected, panel_str = adapter.select_objects([image_uuid], "image")
    assert selected == [image_uuid]
    assert panel_str == "image"
    image_panel.objview.select_objects.assert_called_once_with([1])

    assert adapter.update_metadata(signal_uuid, {"title": "Renamed"}) == signal_uuid
    assert adapter.get_object(signal_uuid).title == "Renamed"
    replacement = _make_signal("Ignored payload title", n=20)
    replacement.set_metadata_option(PROCESSING_PARAMETERS_OPTION, processing_parameters)
    assert adapter.set_object(signal_uuid, replacement) == signal_uuid
    updated = adapter.get_object(signal_uuid)
    assert updated.title == "Renamed"
    assert updated.y.shape == (20,)
    assert get_uuid(updated) == signal_uuid
    assert updated.get_metadata_option(PROCESSING_PARAMETERS_OPTION) == (
        processing_parameters
    )
    assert signal_panel.SIG_OBJECT_MODIFIED.emit.call_count == 2

    adapter.remove_object(image_uuid)
    assert image not in image_panel.objmodel
    image_panel.SIG_OBJECT_REMOVED.emit.assert_called_once_with()


# =============================================================================
# Serialization helpers — _to_python_scalar / _make_json_serializable
# =============================================================================


class TestToPythonScalar:
    """Tests for the ``_to_python_scalar`` private helper."""

    def test_numpy_integer(self) -> None:
        """NumPy integer scalars are converted to ``int``."""
        result = _to_python_scalar(np.int64(7))
        assert result == 7
        assert result.__class__ is int

    def test_numpy_floating(self) -> None:
        """NumPy floating scalars are converted to ``float``."""
        result = _to_python_scalar(np.float32(1.5))
        assert result == pytest.approx(1.5)
        assert result.__class__ is float

    def test_numpy_bool(self) -> None:
        """NumPy boolean scalars are converted to ``bool``."""
        result = _to_python_scalar(np.bool_(True))
        assert result is True
        assert result.__class__ is bool

    def test_passthrough(self) -> None:
        """Native Python values are returned unchanged."""
        assert _to_python_scalar("hello") == "hello"
        assert _to_python_scalar(None) is None
        assert _to_python_scalar(3.14) == pytest.approx(3.14)


class TestMakeJsonSerializable:
    """Tests for the ``_make_json_serializable`` recursive helper."""

    def test_nested_dict_with_numpy(self) -> None:
        """Numpy values nested in dicts are recursively converted."""
        data = {
            "outer": {
                "ints": np.int32(2),
                "floats": np.float64(2.5),
                "bools": np.bool_(False),
                "arr": np.array([1, 2, 3]),
            }
        }
        result = _make_json_serializable(data)
        assert result == {
            "outer": {
                "ints": 2,
                "floats": 2.5,
                "bools": False,
                "arr": [1, 2, 3],
            }
        }
        # Round-trip through JSON to confirm true serialisability:
        json.dumps(result)

    def test_tuple_becomes_list(self) -> None:
        """Tuples are converted to lists for JSON compatibility."""
        result = _make_json_serializable((1, np.float64(2.0), "x"))
        assert result == [1, 2.0, "x"]

    def test_list_with_arrays(self) -> None:
        """Lists of numpy arrays/scalars are recursively converted."""
        result = _make_json_serializable([np.array([1, 2]), np.int8(5)])
        assert result == [[1, 2], 5]

    def test_passthrough_basic_types(self) -> None:
        """Basic JSON-compatible values are returned unchanged."""
        assert _make_json_serializable(42) == 42
        assert _make_json_serializable("abc") == "abc"
        assert _make_json_serializable(None) is None


class TestDeserializeObjMetadata:
    """Tests for ``_deserialize_obj_metadata`` (coords → ndarray restoration)."""

    def test_coords_restored_to_array(self) -> None:
        """Geometry ``coords`` lists are restored to numpy arrays."""
        serialized = {"Geometry_circle": {"coords": [[1.0, 2.0], [3.0, 4.0]]}}
        result = _deserialize_obj_metadata(serialized)
        coords = result["Geometry_circle"]["coords"]
        assert isinstance(coords, np.ndarray)
        np.testing.assert_array_equal(coords, [[1.0, 2.0], [3.0, 4.0]])

    def test_non_coords_list_kept_as_list(self) -> None:
        """Non-``coords`` list entries remain Python lists."""
        result = _deserialize_obj_metadata({"labels": ["a", "b"]})
        assert result == {"labels": ["a", "b"]}
        assert isinstance(result["labels"], list)

    def test_scalar_passthrough(self) -> None:
        """Scalar metadata values are returned unchanged."""
        result = _deserialize_obj_metadata({"title": "T", "count": 7})
        assert result == {"title": "T", "count": 7}


# =============================================================================
# Serialization end-to-end edge cases
# =============================================================================


class TestSerializationEdgeCases:
    """Edge cases for ``serialize_object_to_npz`` / ``deserialize_object_from_npz``."""

    def test_signal_compress_false(self) -> None:
        """``compress=False`` keeps the inner ``y.npy`` uncompressed."""
        obj = _make_signal()
        data = serialize_object_to_npz(obj, compress=False)
        # Inspect ZIP compression mode of an inner entry
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            info = zf.getinfo("y.npy")
            assert info.compress_type == zipfile.ZIP_STORED
        result = deserialize_object_from_npz(data)
        np.testing.assert_array_equal(result.y, obj.y)

    def test_image_compress_false(self) -> None:
        """``compress=False`` keeps the inner ``data.npy`` uncompressed."""
        obj = _make_image()
        data = serialize_object_to_npz(obj, compress=False)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            info = zf.getinfo("data.npy")
            assert info.compress_type == zipfile.ZIP_STORED
        result = deserialize_object_from_npz(data)
        np.testing.assert_array_equal(result.data, obj.data)

    def test_unsupported_object_type_raises(self) -> None:
        """Serialising an unsupported object raises ``TypeError``."""

        class NotASupportedObj:  # pylint: disable=too-few-public-methods
            """Dummy class that is neither SignalObj nor ImageObj."""

        with pytest.raises(TypeError, match="Unsupported object type"):
            serialize_object_to_npz(NotASupportedObj())  # type: ignore[arg-type]

    def test_object_to_metadata_unsupported_type(self) -> None:
        """``object_to_metadata`` rejects unsupported object types."""

        class NotASupportedObj:  # pylint: disable=too-few-public-methods
            """Dummy class."""

        with pytest.raises(TypeError, match="Unsupported object type"):
            object_to_metadata(NotASupportedObj(), "x")  # type: ignore[arg-type]

    def test_deserialize_missing_metadata_raises(self) -> None:
        """NPZ archive without ``metadata.json`` raises ``ValueError``."""
        # NPZ archive without metadata.json
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("x.npy", b"junk")
        with pytest.raises(ValueError, match="missing metadata.json"):
            deserialize_object_from_npz(buf.getvalue())

    def test_deserialize_unknown_type_raises(self) -> None:
        """An unknown ``type`` field in metadata raises ``ValueError``."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", json.dumps({"type": "weird"}))
        with pytest.raises(ValueError, match="Unknown object type"):
            deserialize_object_from_npz(buf.getvalue())

    def test_deserialize_signal_missing_arrays(self) -> None:
        """A signal NPZ without ``x.npy``/``y.npy`` raises ``ValueError``."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", json.dumps({"type": "signal"}))
        with pytest.raises(ValueError, match="missing x.npy or y.npy"):
            deserialize_object_from_npz(buf.getvalue())

    def test_deserialize_image_missing_data(self) -> None:
        """An image NPZ without ``data.npy`` raises ``ValueError``."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("metadata.json", json.dumps({"type": "image"}))
        with pytest.raises(ValueError, match="missing data.npy"):
            deserialize_object_from_npz(buf.getvalue())

    def test_signal_obj_metadata_round_trip(self) -> None:
        """obj.metadata containing numpy scalars/arrays survives a round trip."""
        obj = _make_signal()
        obj.metadata = {
            "Geometry_circle": {"coords": np.array([[0.0, 0.0], [1.0, 1.0]])},
            "scalar_int": np.int64(3),
            "scalar_float": np.float32(1.25),
            "label": "ROI-1",
        }
        data = serialize_object_to_npz(obj)
        result = deserialize_object_from_npz(data)
        # Coords field becomes ndarray, others stay native python types
        assert isinstance(result.metadata["Geometry_circle"]["coords"], np.ndarray)
        np.testing.assert_array_equal(
            result.metadata["Geometry_circle"]["coords"],
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        )
        assert result.metadata["scalar_int"] == 3
        assert result.metadata["scalar_float"] == pytest.approx(1.25)
        assert result.metadata["label"] == "ROI-1"

    def test_image_obj_metadata_with_numpy_origin(self) -> None:
        """Image with numpy-typed x0/y0/dx/dy round-trips to python scalars."""
        obj = _make_image()
        obj.x0 = np.float64(1.5)
        obj.y0 = np.float64(2.5)
        obj.dx = np.float64(0.1)
        obj.dy = np.float64(0.2)
        data = serialize_object_to_npz(obj)
        # Confirm metadata.json has python scalars (json.dumps would otherwise fail)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            meta = json.loads(zf.read("metadata.json"))
            assert isinstance(meta["x0"], float)
            assert isinstance(meta["dx"], float)
        result = deserialize_object_from_npz(data)
        assert result.x0 == pytest.approx(1.5)
        assert result.dx == pytest.approx(0.1)


# =============================================================================
# Routes integration tests
# =============================================================================


class FullMockAdapter:
    """In-memory adapter exposing the full WorkspaceAdapter interface used by routes."""

    def __init__(self) -> None:
        self._objects: dict[str, DataObject] = {}

    # --- queries ------------------------------------------------------------
    def list_objects(self) -> list[tuple[str, str, str]]:
        """Return ``(uuid, name, type)`` tuples for every stored object."""
        return [
            (
                get_uuid(obj),
                name,
                "signal" if isinstance(obj, SignalObj) else "image",
            )
            for name, obj in self._objects.items()
        ]

    def get_object(self, name: str) -> DataObject:
        """Return the stored object for ``name`` or its UUID."""
        if name in self._objects:
            return self._objects[name]
        for obj in self._objects.values():
            if get_uuid(obj) == name:
                return obj
        raise KeyError(f"Object '{name}' not found")

    def object_exists(self, name: str) -> bool:
        """Return whether an object named ``name`` is stored."""
        try:
            self.get_object(name)
            return True
        except KeyError:
            return False

    # --- mutations ----------------------------------------------------------
    def add_object(self, obj: DataObject, overwrite: bool = False) -> None:
        """Store ``obj`` under its ``title``; reject duplicates without overwrite."""
        name = obj.title
        if name in self._objects and not overwrite:
            raise ValueError(f"Object '{name}' already exists")
        self._objects[name] = obj

    def remove_object(self, name: str) -> None:
        """Remove the object stored under ``name`` or raise ``KeyError``."""
        obj = self.get_object(name)
        stored_name = next(key for key, value in self._objects.items() if value is obj)
        del self._objects[stored_name]

    def update_metadata(self, name: str, metadata: dict) -> str:
        """Update non-``None`` attributes of the object stored under ``name``."""
        obj = self.get_object(name)
        stored_name = next(key for key, value in self._objects.items() if value is obj)
        for k, v in metadata.items():
            if v is not None and hasattr(obj, k):
                setattr(obj, k, v)
        if obj.title != stored_name:
            del self._objects[stored_name]
            self._objects[obj.title] = obj
        return get_uuid(obj)

    def set_object(self, name: str, obj: DataObject) -> str:
        """Replace the data of the object stored under ``name``."""
        existing = self.get_object(name)
        stored_name = next(
            key for key, value in self._objects.items() if value is existing
        )
        object_uuid = get_uuid(existing)
        obj.title = existing.title
        obj.set_metadata_option("uuid", object_uuid)
        self._objects[stored_name] = obj
        return object_uuid


@pytest.fixture(name="api_client")
def _api_client():
    """Build a FastAPI TestClient bound to a fresh ``FullMockAdapter``."""
    app = FastAPI()
    app.include_router(router)

    adapter = FullMockAdapter()
    token = "test-token-route-12345"
    set_adapter(adapter)
    set_auth_token(token)
    set_server_url("http://localhost:8765")
    set_localhost_no_token(False)

    yield TestClient(app), token, adapter

    # Reset module-level state so the next test starts clean
    set_localhost_no_token(False)


def _auth(token: str) -> dict:
    """Return an ``Authorization`` header dict for the given bearer token."""
    return {"Authorization": f"Bearer {token}"}


class TestAuthBranches:
    """Authentication branches in ``verify_token`` not covered by webapi_test.py."""

    def test_localhost_bypass(self, api_client) -> None:
        """Localhost clients bypass token verification when configured."""
        _client, _token, adapter = api_client
        adapter.add_object(_make_signal("S1"))
        # Build a dedicated TestClient that advertises a real localhost client IP
        # (the default starlette TestClient host is ``testclient`` and would not
        # match the localhost-bypass condition in ``verify_token``).
        app = FastAPI()
        app.include_router(router)
        local_client = TestClient(app, client=("127.0.0.1", 50000))
        set_localhost_no_token(True)
        try:
            response = local_client.get("/api/v1/objects")
            assert response.status_code == 200
            assert response.json()["count"] == 1
        finally:
            set_localhost_no_token(False)


class TestObjectListing:
    """``GET /objects`` and ``GET /objects/{name}`` endpoints."""

    def test_list_with_signal_and_image(self, api_client) -> None:
        """``GET /objects`` lists both signal and image objects."""
        client, token, adapter = api_client
        adapter.add_object(_make_signal("S1"))
        adapter.add_object(_make_image("I1"))
        response = client.get("/api/v1/objects", headers=_auth(token))
        assert response.status_code == 200
        names = {o["name"] for o in response.json()["objects"]}
        assert names == {"S1", "I1"}

    def test_metadata_known_object(self, api_client) -> None:
        """``GET /objects/{name}`` returns metadata for a known object."""
        client, token, adapter = api_client
        adapter.add_object(_make_signal("S1"))
        response = client.get("/api/v1/objects/S1", headers=_auth(token))
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "S1"
        assert body["type"] == "signal"

    def test_metadata_by_uuid_returns_title_as_name(self, api_client) -> None:
        """``GET`` by UUID keeps title and UUID response fields distinct."""
        client, token, adapter = api_client
        obj = _make_signal("S1")
        adapter.add_object(obj)
        object_uuid = get_uuid(obj)
        response = client.get(f"/api/v1/objects/{object_uuid}", headers=_auth(token))
        assert response.status_code == 200
        assert response.json()["name"] == "S1"
        assert response.json()["uuid"] == object_uuid

    def test_metadata_missing_object(self, api_client) -> None:
        """``GET /objects/{name}`` returns 404 for an unknown object."""
        client, token, _adapter = api_client
        response = client.get("/api/v1/objects/nope", headers=_auth(token))
        assert response.status_code == 404


class TestPatchAndDelete:
    """``PATCH /objects/{name}/metadata`` and ``DELETE /objects/{name}``."""

    def test_patch_updates_known_field(self, api_client) -> None:
        """``PATCH`` updates supported metadata fields on an existing object."""
        client, token, adapter = api_client
        adapter.add_object(_make_signal("S1"))
        response = client.patch(
            "/api/v1/objects/S1/metadata",
            json={"title": "S1", "xlabel": "Time", "ylabel": "Volts"},
            headers=_auth(token),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["xlabel"] == "Time"
        assert body["ylabel"] == "Volts"

    def test_patch_rename_returns_updated_metadata(self, api_client) -> None:
        """``PATCH`` reloads a renamed object through its stable UUID."""
        client, token, adapter = api_client
        adapter.add_object(_make_signal("Old"))
        response = client.patch(
            "/api/v1/objects/Old/metadata",
            json={"title": "New"},
            headers=_auth(token),
        )
        assert response.status_code == 200
        assert response.json()["name"] == "New"
        assert adapter.object_exists("New")

    def test_patch_missing_object(self, api_client) -> None:
        """``PATCH`` on an unknown object returns 404."""
        client, token, _adapter = api_client
        response = client.patch(
            "/api/v1/objects/missing/metadata",
            json={"xlabel": "X"},
            headers=_auth(token),
        )
        assert response.status_code == 404

    def test_delete_known(self, api_client) -> None:
        """``DELETE`` removes an existing object."""
        client, token, adapter = api_client
        adapter.add_object(_make_signal("S1"))
        response = client.delete("/api/v1/objects/S1", headers=_auth(token))
        assert response.status_code == 204
        assert not adapter.object_exists("S1")

    def test_delete_by_uuid(self, api_client) -> None:
        """``DELETE`` removes an object addressed by full UUID."""
        client, token, adapter = api_client
        obj = _make_signal("S1")
        adapter.add_object(obj)
        response = client.delete(
            f"/api/v1/objects/{get_uuid(obj)}", headers=_auth(token)
        )
        assert response.status_code == 204
        assert not adapter.object_exists("S1")

    def test_delete_missing(self, api_client) -> None:
        """``DELETE`` on an unknown object returns 404."""
        client, token, _adapter = api_client
        response = client.delete("/api/v1/objects/missing", headers=_auth(token))
        assert response.status_code == 404


class TestBinaryDataEndpoints:
    """``GET /objects/{name}/data`` and ``PUT /objects/{name}/data`` endpoints."""

    def test_get_signal_data_returns_npz(self, api_client) -> None:
        """``GET /objects/{name}/data`` returns an NPZ payload for a signal."""
        client, token, adapter = api_client
        obj = _make_signal("S1")
        adapter.add_object(obj)
        response = client.get("/api/v1/objects/S1/data", headers=_auth(token))
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-npz"
        assert 'filename="S1.npz"' in response.headers["content-disposition"]
        # Round-trip the body
        result = deserialize_object_from_npz(response.content)
        np.testing.assert_array_equal(result.y, obj.y)

    def test_get_data_unicode_filename(self, api_client) -> None:
        """Non-ASCII names use RFC 5987 ``filename*`` encoding."""
        client, token, adapter = api_client
        adapter.add_object(_make_signal("éàü"))
        response = client.get("/api/v1/objects/éàü/data", headers=_auth(token))
        assert response.status_code == 200
        # RFC 5987 encoding kicks in for non-ASCII names
        assert "filename*=UTF-8''" in response.headers["content-disposition"]

    def test_get_data_compress_false(self, api_client) -> None:
        """``compress=false`` yields an NPZ archive with stored entries."""
        client, token, adapter = api_client
        adapter.add_object(_make_image("I1"))
        response = client.get(
            "/api/v1/objects/I1/data?compress=false", headers=_auth(token)
        )
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            assert zf.getinfo("data.npy").compress_type == zipfile.ZIP_STORED

    def test_get_data_missing_object(self, api_client) -> None:
        """``GET data`` on an unknown object returns 404."""
        client, token, _adapter = api_client
        response = client.get("/api/v1/objects/nope/data", headers=_auth(token))
        assert response.status_code == 404

    def test_put_creates_object(self, api_client) -> None:
        """``PUT data`` creates a new object from an NPZ payload."""
        client, token, adapter = api_client
        payload = serialize_object_to_npz(_make_signal("New"))
        response = client.put(
            "/api/v1/objects/New/data",
            content=payload,
            headers={**_auth(token), "Content-Type": "application/x-npz"},
        )
        assert response.status_code == 201
        assert adapter.object_exists("New")

    def test_put_conflict_without_overwrite(self, api_client) -> None:
        """``PUT data`` without ``overwrite=true`` returns 409 on conflict."""
        client, token, adapter = api_client
        adapter.add_object(_make_signal("Dup"))
        payload = serialize_object_to_npz(_make_signal("Dup"))
        response = client.put(
            "/api/v1/objects/Dup/data",
            content=payload,
            headers={**_auth(token), "Content-Type": "application/x-npz"},
        )
        assert response.status_code == 409

    def test_put_overwrite(self, api_client) -> None:
        """``PUT data?overwrite=true`` replaces the existing object."""
        client, token, adapter = api_client
        adapter.add_object(_make_signal("Dup"))
        payload = serialize_object_to_npz(_make_signal("Dup"))
        response = client.put(
            "/api/v1/objects/Dup/data?overwrite=true",
            content=payload,
            headers={**_auth(token), "Content-Type": "application/x-npz"},
        )
        assert response.status_code == 201

    def test_put_invalid_npz_returns_422(self, api_client) -> None:
        """``PUT data`` with a non-NPZ payload returns a client/server error."""
        client, token, _adapter = api_client
        response = client.put(
            "/api/v1/objects/Bad/data",
            content=b"not a zip archive",
            headers={**_auth(token), "Content-Type": "application/x-npz"},
        )
        # zipfile fails before the value-error branch — should produce 5xx,
        # but the route classifies any non-HTTPException as 500 (with detail).
        assert response.status_code in (422, 500)

    def test_set_object_missing(self, api_client) -> None:
        """``PUT /objects/{name}`` returns 404 on an unknown object."""
        client, token, _adapter = api_client
        payload = serialize_object_to_npz(_make_signal("Ghost"))
        response = client.put(
            "/api/v1/objects/Ghost",
            content=payload,
            headers={**_auth(token), "Content-Type": "application/x-npz"},
        )
        assert response.status_code == 404

    def test_set_object_in_place(self, api_client) -> None:
        """``PUT /objects/{name}`` updates the existing object in place."""
        client, token, adapter = api_client
        adapter.add_object(_make_signal("Live", n=10))
        new_payload = serialize_object_to_npz(_make_signal("Live", n=20))
        response = client.put(
            "/api/v1/objects/Live",
            content=new_payload,
            headers={**_auth(token), "Content-Type": "application/x-npz"},
        )
        assert response.status_code == 200
        # Adapter now has the updated object
        assert adapter.get_object("Live").y.shape == (20,)

    def test_set_object_by_uuid(self, api_client) -> None:
        """``PUT /objects/{uuid}`` preserves title and stable identity."""
        client, token, adapter = api_client
        original = _make_signal("Live", n=10)
        adapter.add_object(original)
        object_uuid = get_uuid(original)
        payload = serialize_object_to_npz(_make_signal("Payload", n=20))
        response = client.put(
            f"/api/v1/objects/{object_uuid}",
            content=payload,
            headers={**_auth(token), "Content-Type": "application/x-npz"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Live"
        assert response.json()["uuid"] == object_uuid
        assert adapter.get_object(object_uuid).y.shape == (20,)


if __name__ == "__main__":
    pytest.main([__file__])
