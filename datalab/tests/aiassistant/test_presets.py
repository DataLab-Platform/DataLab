# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for :mod:`datalab.aiassistant.presets`."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

from datalab.aiassistant.presets import (
    BASE_URL_PRESETS,
    BaseUrlPreset,
    detect_preset,
)
from datalab.aiassistant.presets import test_connection as probe_connection


def test_presets_contain_known_endpoints() -> None:
    """Built-in presets cover the documented OpenAI-compatible endpoints."""
    keys = {p.key for p in BASE_URL_PRESETS}
    assert keys == {"openai", "ollama", "lmstudio", "llamacpp", "vllm"}
    for preset in BASE_URL_PRESETS:
        assert preset.base_url.startswith(("http://", "https://"))
        assert preset.default_model


def test_detect_preset_matches_known_url() -> None:
    """A registered URL maps back to its preset (case + trailing slash insensitive)."""
    assert detect_preset("http://localhost:11434/v1").key == "ollama"
    assert detect_preset("HTTP://LOCALHOST:11434/V1/").key == "ollama"


def test_detect_preset_empty_returns_openai() -> None:
    """An empty/None URL is treated as the OpenAI cloud default."""
    assert detect_preset("").key == "openai"
    assert detect_preset(None).key == "openai"


def test_detect_preset_unknown_returns_none() -> None:
    """A custom URL with no match returns ``None``."""
    assert detect_preset("https://example.com/v1") is None


class _FakeResp:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_test_connection_ok_counts_models() -> None:
    """Successful probe reports latency and parses the standard OpenAI model list."""
    payload = json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]}).encode("utf-8")
    with mock.patch(
        "datalab.aiassistant.presets.urllib.request.urlopen",
        return_value=_FakeResp(payload),
    ):
        result = probe_connection("http://localhost:11434/v1", api_key=None)
    assert result.ok is True
    assert result.model_count == 2
    assert result.latency_ms is not None
    assert result.status_code == 200


def test_test_connection_sends_bearer_when_api_key_provided() -> None:
    """The ``Authorization`` header is set only when an API key is given."""
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["headers"] = dict(req.headers)
        captured["url"] = req.full_url
        return _FakeResp(b"{}")

    with mock.patch(
        "datalab.aiassistant.presets.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        probe_connection("https://api.openai.com/v1", api_key="sk-test")
    assert captured["url"].endswith("/models")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_test_connection_no_header_without_key() -> None:
    """Without an API key, no ``Authorization`` header is emitted."""
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):  # noqa: ARG001
        captured["headers"] = dict(req.headers)
        return _FakeResp(b"{}")

    with mock.patch(
        "datalab.aiassistant.presets.urllib.request.urlopen", side_effect=fake_urlopen
    ):
        probe_connection("http://localhost:11434/v1", api_key="")
    assert "Authorization" not in captured["headers"]


def test_test_connection_http_error_reports_status() -> None:
    """An HTTPError is wrapped into a non-OK result carrying the status code."""
    err = urllib.error.HTTPError(
        url="x", code=401, msg="unauthorized", hdrs=None, fp=io.BytesIO(b"bad key")
    )
    with mock.patch(
        "datalab.aiassistant.presets.urllib.request.urlopen", side_effect=err
    ):
        result = probe_connection("https://api.openai.com/v1", api_key="bad")
    assert result.ok is False
    assert result.status_code == 401
    assert "401" in result.message


def test_test_connection_network_error_reports_reason() -> None:
    """A URLError surfaces as a non-OK result mentioning the reason."""
    with mock.patch(
        "datalab.aiassistant.presets.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        result = probe_connection("http://localhost:11434/v1", api_key=None)
    assert result.ok is False
    assert "connection refused" in result.message


def test_dataclass_preset_is_immutable() -> None:
    """``BaseUrlPreset`` is a frozen dataclass."""
    preset = BASE_URL_PRESETS[0]
    assert isinstance(preset, BaseUrlPreset)
    try:
        preset.key = "x"  # type: ignore[misc]
    except Exception:  # dataclasses.FrozenInstanceError
        return
    raise AssertionError("Preset should be immutable")
