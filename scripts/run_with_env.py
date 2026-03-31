# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Run a command with environment variables loaded from a .env file."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def load_env_file(env_path: str | None = None) -> None:
    """Load environment variables from a .env file."""
    # Set a flag to indicate that the environment has been loaded by this script
    # This prevents batch scripts (like utils.bat) from reloading .env and overwriting
    # variables
    os.environ["DATALAB_ENV_LOADED"] = "1"

    if env_path is None:
        # Get ".env" file from the current directory
        env_path = Path.cwd() / ".env"
    if not Path(env_path).is_file():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    print(f"Loading environment variables from: {env_path}")
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = os.path.expandvars(value.strip())

            # Handle PATH variable specifically:
            # 1. Convert relative paths to absolute paths
            # 2. Normalize path separators
            if key.strip().upper() == "PATH":
                paths = value.split(os.pathsep)
                abs_paths = []
                for p in paths:
                    p = p.strip()
                    if not p:
                        continue
                    # Check if it looks like a relative path component
                    # (not starting with drive or root)
                    # Note: This simple check assumes standard usage in .env
                    if not os.path.isabs(p) and not p.startswith("%"):
                        try:
                            # Resolve relative to .env file directory
                            p = str((Path(env_path).parent / p).resolve())
                        except Exception:
                            pass  # Keep as is if resolution fails
                    abs_paths.append(os.path.normpath(p))
                value = os.pathsep.join(abs_paths)

            os.environ[key.strip()] = value
            print(f"  Loaded variable: {key.strip()}={value}")


def execute_command(command: list[str]) -> int:
    """Execute a command with the loaded environment variables."""
    print("Executing command:")
    print(" ".join(command))
    print("")
    result = subprocess.call(command)
    print(f"Process exited with code {result}")
    return result


def main() -> None:
    """Main function to load environment variables and execute a command."""
    if len(sys.argv) < 2:
        print("Usage: python run_with_env.py <command> [args ...]")
        sys.exit(1)
    print("🏃 Running with environment variables")
    load_env_file()
    return execute_command(sys.argv[1:])


if __name__ == "__main__":
    main()
