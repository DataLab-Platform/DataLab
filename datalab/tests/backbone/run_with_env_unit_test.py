# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests of ``scripts/run_with_env.py``, the wrapper of the VS Code tasks

The wrapper must exit with the exit code of the command it runs: otherwise VS Code
reports failed tasks as successful, and runs the tasks chained after them (see
issue #552).
"""

from __future__ import annotations

import os
import os.path as osp
import subprocess
import sys

import pytest

SCRIPT_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)), "..", "..", "..", "scripts", "run_with_env.py"
)


@pytest.mark.skipif(
    not osp.isfile(SCRIPT_PATH),
    reason="scripts/run_with_env.py is not available (installed package)",
)
@pytest.mark.parametrize("exit_code", [0, 3])
def test_exit_code(tmp_path, exit_code: int) -> None:
    """The wrapper exits with the exit code of the command it runs"""
    (tmp_path / ".env").write_text("", encoding="utf-8")
    # Without interpreter selection variables, the wrapper runs the command with
    # the interpreter running it (no virtual environment in the temporary folder)
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("PYTHON", "WINPYDIRBASE", "VENV_DIR")
    }
    env["PYTHONIOENCODING"] = "utf-8"  # The wrapper prints emoji
    command = f"import sys; sys.exit({exit_code})"
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "python", "-c", command],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert f"Process exited with code {exit_code}" in result.stdout, result.stdout
    assert result.returncode == exit_code
