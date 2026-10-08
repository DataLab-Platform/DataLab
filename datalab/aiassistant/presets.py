# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Base URL presets and connection probe for OpenAI-compatible providers.

Ported from the DataLab-Web ``src/aiassistant/settings.ts`` helpers, adapted
for synchronous :mod:`urllib` usage.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaseUrlPreset:
    """A named OpenAI-compatible endpoint preset.

    Args:
        key: Stable identifier (used to detect a preset from a base URL).
        label: Short human-readable label (translated at display time).
        base_url: Default ``base_url`` value to pre-fill.
        default_model: Suggested default model name.
        is_local: Whether the endpoint targets a local server.
    """

    key: str
    label: str
    base_url: str
    default_model: str
    is_local: bool


BASE_URL_PRESETS: tuple[BaseUrlPreset, ...] = (
    BaseUrlPreset(
        key="openai",
        label="OpenAI (cloud)",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        is_local=False,
    ),
    BaseUrlPreset(
        key="ollama",
        label="Ollama (local)",
        base_url="http://localhost:11434/v1",
        default_model="llama3.2:3b",
        is_local=True,
    ),
    BaseUrlPreset(
        key="lmstudio",
        label="LM Studio (local)",
        base_url="http://localhost:1234/v1",
        default_model="local-model",
        is_local=True,
    ),
    BaseUrlPreset(
        key="llamacpp",
        label="llama.cpp server",
        base_url="http://localhost:8080/v1",
        default_model="local-model",
        is_local=True,
    ),
    BaseUrlPreset(
        key="vllm",
        label="vLLM (local)",
        base_url="http://localhost:8000/v1",
        default_model="local-model",
        is_local=True,
    ),
)


def detect_preset(base_url: str | None) -> BaseUrlPreset | None:
    """Return the preset matching ``base_url`` (case-insensitive), if any.

    An empty or ``None`` URL matches the OpenAI cloud preset.
    """
    if not base_url:
        return BASE_URL_PRESETS[0]
    needle = base_url.strip().rstrip("/").lower()
    for preset in BASE_URL_PRESETS:
        if preset.base_url.rstrip("/").lower() == needle:
            return preset
    return None


@dataclass(frozen=True)
class ConnectionProbeResult:
    """Outcome of :func:`test_connection`.

    Args:
        ok: ``True`` if the endpoint responded with a 2xx status.
        message: Short human-readable summary.
        latency_ms: Round-trip latency in milliseconds (``None`` on failure).
        model_count: Number of models reported by the endpoint, if known.
        status_code: HTTP status code when the request reached the server.
        details: Verbose diagnostic text (e.g. raw error body) for the dialog's
         expandable details panel. ``None`` when nothing useful to show.
    """

    ok: bool
    message: str
    latency_ms: float | None = None
    model_count: int | None = None
    status_code: int | None = None
    details: str | None = None


def _models_url(base_url: str | None) -> str:
    return (base_url or BASE_URL_PRESETS[0].base_url).rstrip("/") + "/models"


def test_connection(
    base_url: str | None,
    api_key: str | None,
    timeout: float = 5.0,
) -> ConnectionProbeResult:
    """Probe ``GET {base_url}/models`` to validate URL, auth and reachability.

    Args:
        base_url: OpenAI-compatible base URL (``None``/empty → OpenAI cloud).
        api_key: Bearer token (sent only when non-empty).
        timeout: Request timeout in seconds.

    Returns:
        A :class:`ConnectionProbeResult` describing the outcome. Never raises.
    """
    url = _models_url(base_url)
    headers: dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, method="GET", headers=headers)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            status = getattr(resp, "status", 200)
            raw = resp.read()
        model_count = _count_models(raw)
        suffix = f" — {model_count} model(s)" if model_count is not None else ""
        return ConnectionProbeResult(
            ok=True,
            message=f"OK ({elapsed_ms:.0f} ms){suffix}",
            latency_ms=elapsed_ms,
            model_count=model_count,
            status_code=status,
        )
    except urllib.error.HTTPError as exc:
        try:
            raw_body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # pylint: disable=broad-except
            raw_body = ""
        summary = _extract_error_message(raw_body) or exc.reason or "request failed"
        return ConnectionProbeResult(
            ok=False,
            message=f"HTTP {exc.code} — {summary}",
            status_code=exc.code,
            details=raw_body.strip() or None,
        )
    except urllib.error.URLError as exc:
        return ConnectionProbeResult(ok=False, message=f"Network error: {exc.reason}")
    except Exception as exc:  # pylint: disable=broad-except
        return ConnectionProbeResult(ok=False, message=f"Request failed: {exc}")


def _extract_error_message(raw: str) -> str | None:
    """Backward-compatible alias for :func:`extract_error_message`."""
    # pylint: disable-next=import-outside-toplevel
    from datalab.aiassistant.providers.base import extract_error_message

    return extract_error_message(raw)


def _count_models(raw: bytes) -> int | None:
    """Return the number of models in an OpenAI ``/models`` response, if parsable."""
    try:
        data: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(data, dict):
        items = data.get("data")
        if isinstance(items, list):
            return len(items)
    if isinstance(data, list):
        return len(data)
    return None
