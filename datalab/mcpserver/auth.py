# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Token of the MCP server.

By default, a new token is created at each start of the server. With the
persistent token option, the token is kept in a dedicated file of the DataLab
configuration directory, so that the configuration of the external assistants
does not change between sessions. The token is never logged.
"""

from __future__ import annotations

import os
import secrets

from datalab.config import get_config_path

TOKEN_FILE_NAME = "mcp_token"


def get_token_file_path() -> str:
    """Return the path of the file storing the persistent token."""
    return get_config_path(TOKEN_FILE_NAME)


def generate_token() -> str:
    """Return a new random token."""
    return secrets.token_urlsafe(32)


def _write_token_file(token: str) -> None:
    """Write the persistent token, readable by the current user only."""
    path = get_token_file_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(token)


def get_token(persistent: bool) -> str:
    """Return the token of the server.

    Args:
        persistent: If True, return the persistent token (created if needed).
         Otherwise, return a new session token and delete any persistent token.
    """
    if not persistent:
        delete_persistent_token()
        return generate_token()
    try:
        with open(get_token_file_path(), encoding="utf-8") as file:
            token = file.read().strip()
    except OSError:
        token = ""
    if not token:
        token = generate_token()
        _write_token_file(token)
    return token


def renew_token(persistent: bool) -> str:
    """Return a new token, replacing the persistent token if any.

    Args:
        persistent: If True, store the new token as the persistent token.
    """
    token = generate_token()
    if persistent:
        _write_token_file(token)
    return token


def delete_persistent_token() -> None:
    """Delete the persistent token, if any."""
    try:
        os.remove(get_token_file_path())
    except FileNotFoundError:
        pass
