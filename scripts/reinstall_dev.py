# script/reinstall_dev.py
"""
Reinstall multiple local libraries in editable mode for development.

Workflow:
  1) Try to uninstall all target libraries in one command (ignore errors if some are not installed).
  2) Reinstall each library in editable mode from a sibling folder: ../<library>.

This script uses the same Python interpreter that runs it (sys.executable),
so pip operations happen in the same environment (e.g., your active venv).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from typing import Iterable

# Absolute path to the Python executable that runs this script.
# Using sys.executable ensures pip targets the same environment (e.g., your venv).
PY = sys.executable


def run(cmd: list[str]) -> None:
    """
    Echo and execute a subprocess command.
    Raises:
        subprocess.CalledProcessError: if the command returns a non-zero exit code.
    """
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def uninstall_many(packages: Iterable[str]) -> None:
    """
    Attempt to uninstall multiple packages at once.
    If uninstall fails (e.g., a package is not installed), we keep going.

    Args:
        packages: An iterable of package names (pip distribution names).
    """
    pkgs = list(packages)
    if not pkgs:
        return
    try:
        run([PY, "-m", "pip", "uninstall", "-y", *pkgs])
    except subprocess.CalledProcessError as e:
        # Continue even if the uninstall step fails (for one or more packages)
        print(f"[WARN] Uninstall returned {e.returncode} — continuing...", flush=True)


def install_editable_many(packages: Iterable[str]) -> None:
    """
    Install each package in editable mode from ../<package_dir>.

    Assumes your project layout has sibling folders one level up, e.g.:
      ../guidata
      ../plotpy
      ../sigima

    Args:
        packages: An iterable of package names (also used as directory names).
    """
    for pkg in packages:
        run([PY, "-m", "pip", "install", "-e", f"../{pkg}"])


def remove_residual_dirs(packages: Iterable[str]) -> None:
    """
    Force remove residual package directories from site-packages.
    This mimics: rm -rf .venv/Lib/site-packages/<pkg>
    to ensure a clean slate before reinstalling.
    """
    # Locates site-packages, e.g. .venv/Lib/site-packages
    site_packages = sysconfig.get_path("purelib")

    for pkg in packages:
        target = os.path.join(site_packages, pkg)
        if os.path.isdir(target):
            print(f"Removing residual directory: {target}", flush=True)
            try:
                shutil.rmtree(target)
            except OSError as e:
                print(f"[WARN] Failed to remove {target}: {e}", flush=True)


def reinstall_packages(packages: list[str]) -> None:
    """
    High-level orchestration for many packages:
      - Uninstall all of them (ignore failures)
      - Force remove residual directories from site-packages
      - Install each in editable mode
    """
    # 1) Uninstall (ignore if not installed)
    uninstall_many(packages)

    # 2) Force remove residual directories (essential for clean reinstall)
    remove_residual_dirs(packages)

    # 3) Editable installs
    install_editable_many(packages)


if __name__ == "__main__":
    # ⭐ Fixed, editable local libraries to manage (edit as needed)
    PACKAGES = ["guidata", "plotpy", "sigima"]

    print("🏃 Reinstalling editable packages:", ", ".join(PACKAGES))
    reinstall_packages(PACKAGES)
