# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Smoke-test the DataLab Windows installer and its standalone executable

Installs the MSI installer in a disposable folder, starts the installed application,
checks the installation and uninstalls it (Windows only, standard library only)::

    python scripts/frozen_smoke_test.py all --msi wix/DataLab-1.4.0.msi
        --dist dist/DataLab --workdir build/msi-smoke-test

Steps of the ``all`` command:

1. refuse to run if DataLab with the same ProductCode (i.e. any DataLab of the same
   major version) is installed on this computer: it would be uninstalled at the end;
2. hash the MSI and the other ``--artifact`` files, which must not be modified;
3. install the MSI silently in ``WORKDIR/install`` (per user, without elevation,
   or for all users with ``--per-machine``);
4. compare the installed files with the PyInstaller folder (``--dist``);
5. start the installed ``DataLab.exe`` in unattended mode with screenshots: it must
   quit by itself with exit code 0, without writing ``datalab_error.log`` or a
   traceback/faulthandler log, after taking at least one screenshot;
6. run ``DataLab.exe --smoke-test`` (see ``datalab/utils/smoketest.py``): every check
   must pass;
7. uninstall the MSI and check that the installed files were removed;
8. check that the hashed files were not modified.

The ``gui`` and ``probe`` commands run steps 5 and 6 on an executable, e.g.
``dist/DataLab/DataLab.exe``. Each run of the executable uses its own user profile
and working directory. Screenshots, reports, logs and ``summary.json`` are collected
in ``WORKDIR/evidence``. The exit code is 0 if every step passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

#: File created in the work directory, to allow its removal by the next run
WORKDIR_MARKER = ".datalab-smoke-test"
#: Log files written by DataLab in its configuration folder (not empty: an error)
APP_LOG_RE = re.compile(r"(traceback|faulthandler).*\.log$", re.IGNORECASE)
#: Environment variables which must not reach the executable
UNSET_VARIABLES = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
#: Environment variables pointing to the user profile
PROFILE_VARIABLES = ("USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA")

# msiexec exit codes
MSI_SUCCESS = (0, 3010)  # 3010: success, a restart is required
MSI_BUSY = 1618  # another installation is in progress
MSI_OTHER_VERSION = 1638  # another version of this product is installed
# MsiQueryProductState results
INSTALLSTATE_UNKNOWN = -1
INSTALLSTATE_DEFAULT = 5


class SmokeTestError(Exception):
    """Failed smoke-test step"""


def log(message: str = "") -> None:
    """Print a message immediately (CI logs)."""
    print(message, flush=True)


def annotate(kind: str, message: str) -> None:
    """Print a GitHub Actions annotation, or a plain message outside CI."""
    message = " ".join(message.split())
    log(f"::{kind}::{message}" if GITHUB_ACTIONS else f"{kind.upper()}: {message}")


def sha256(path: str) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_makewxs():
    """Import ``wix/makewxs.py``, which computes the ProductCode of the MSI."""
    spec = importlib.util.spec_from_file_location(
        "makewxs", osp.join(ROOT, "wix", "makewxs.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_product_state(product_code: str) -> int:
    """Return the Windows Installer state of a product (e.g. -1: not installed)."""
    func = ctypes.WinDLL("msi").MsiQueryProductStateW
    func.argtypes = [ctypes.c_wchar_p]
    func.restype = ctypes.c_int
    return func(product_code)


def run_msiexec(arguments: str, log_path: str) -> int:
    """Run msiexec silently, retrying while another installation is in progress.

    Args:
        arguments: msiexec arguments (quoted for msiexec, which parses its own
         command line: ``PROPERTY="value"``).
        log_path: Path of the verbose log file.

    Returns:
        Exit code of msiexec.
    """
    command = f'msiexec.exe {arguments} /qn /norestart /l*v "{log_path}"'
    for _attempt in range(10):
        log(command)
        code = subprocess.call(command)
        if code != MSI_BUSY:
            return code
        log("Another installation is in progress: retrying in 30 s")
        time.sleep(30)
    return code


def list_files(folder: str) -> dict[str, int]:
    """Return the size of the files of a folder, by relative path."""
    files = {}
    for root, _dirs, names in os.walk(folder):
        for name in names:
            path = osp.join(root, name)
            files[osp.relpath(path, folder)] = osp.getsize(path)
    return files


def prepare_workdir(workdir: str) -> str:
    """Create an empty work directory, and return its evidence folder.

    A work directory left by a previous run is removed. Any other non-empty
    directory is refused, to never remove user files.
    """
    if osp.isdir(workdir) and os.listdir(workdir):
        if not osp.isfile(osp.join(workdir, WORKDIR_MARKER)):
            raise SmokeTestError(
                f"{workdir} is not empty and was not created by this script: "
                "choose another work directory"
            )
        shutil.rmtree(workdir)
    evidence = osp.join(workdir, "evidence")
    os.makedirs(evidence, exist_ok=True)
    with open(osp.join(workdir, WORKDIR_MARKER), "w", encoding="utf-8"):
        pass
    return evidence


class Steps:
    """Run smoke-test steps and record their results.

    Args:
        title: Title of the summary.
        evidence: Folder where ``summary.json`` is written.
    """

    def __init__(self, title: str, evidence: str) -> None:
        self.title = title
        self.evidence = evidence
        self.results: list[dict] = []
        self.warnings: list[str] = []

    @property
    def failed(self) -> bool:
        """True if a step failed."""
        return any(result["status"] == "failed" for result in self.results)

    def run(self, name: str, func: Callable[[], str]) -> None:
        """Run a step, which returns a short description of its result.

        Raises:
            SmokeTestError: The step failed.
        """
        log(f"::group::{name}" if GITHUB_ACTIONS else f"\n=== {name}")
        start = time.monotonic()
        result = {"step": name, "status": "failed", "detail": "interrupted"}
        self.results.append(result)
        try:
            detail = func()
        except Exception as exc:
            if not isinstance(exc, SmokeTestError):
                traceback.print_exc()
            result["detail"] = str(exc)
            raise SmokeTestError(f"{name}: {exc}") from exc
        else:
            result.update(status="passed", detail=detail)
        finally:
            result["duration"] = round(time.monotonic() - start, 1)
            status = "OK" if result["status"] == "passed" else "FAILED"
            log(f"{status}: {result['detail']}")
            if GITHUB_ACTIONS:
                log("::endgroup::")

    def warn(self, message: str) -> None:
        """Record a warning."""
        self.warnings.append(message)
        annotate("warning", message)

    def finish(self, **information: str) -> int:
        """Write the summary and return the exit code."""
        for result in self.results:
            if result["status"] == "failed":
                annotate("error", f"{result['step']}: {result['detail']}")
        summary = {
            **information,
            "result": "failed" if self.failed else "passed",
            "steps": self.results,
            "warnings": self.warnings,
        }
        with open(osp.join(self.evidence, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        lines = [
            f"### {self.title}: {summary['result']}",
            "",
            "| Step | Status | Duration (s) | Detail |",
            "| --- | --- | --- | --- |",
        ]
        for res in self.results:
            detail = " ".join(str(res["detail"]).split()).replace("|", "\\|")
            lines.append(
                f"| {res['step']} | {res['status']} | {res['duration']} | {detail} |"
            )
        lines += [f"\n:warning: {warning}" for warning in self.warnings]
        markdown = "\n".join(lines) + "\n"
        log("\n" + markdown)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
                f.write(markdown)
        return 1 if self.failed else 0


class ExecutableRun:
    """Run the DataLab executable in an isolated user profile and working directory.

    Args:
        exe: Path of ``DataLab.exe``.
        workdir: Work directory (runs are prepared in ``run/NAME``).
        evidence: Folder where logs written by DataLab are copied.
        name: Name of the run.
    """

    def __init__(self, exe: str, workdir: str, evidence: str, name: str) -> None:
        self.exe = exe
        self.evidence = evidence
        self.name = name
        self.cwd = osp.join(workdir, "run", name, "cwd")
        self.profile = osp.join(workdir, "run", name, "profile")
        os.makedirs(self.cwd)
        os.makedirs(self.profile)

    def execute(self, arguments: list[str], timeout: float) -> list[str]:
        """Run the executable and return the problems found.

        Problems: timeout, non-zero exit code, ``datalab_error.log`` written in the
        working directory, non-empty traceback or faulthandler log in the profile.
        """
        env = {k: v for k, v in os.environ.items() if k not in UNSET_VARIABLES}
        env.update(dict.fromkeys(PROFILE_VARIABLES, self.profile))
        command = [self.exe, *arguments]
        log(subprocess.list2cmdline(command))
        start = time.monotonic()
        problems = []
        with subprocess.Popen(command, cwd=self.cwd, env=env) as process:
            try:
                code = process.wait(timeout)
            except subprocess.TimeoutExpired:
                # Also stop the processes started by DataLab (processing pool)
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(process.pid)])
                process.wait()
                problems.append(f"still running after {timeout:g} s (stopped)")
            else:
                log(f"Exit code {code} after {time.monotonic() - start:.1f} s")
                if code != 0:
                    problems.append(f"exit code {code}")
        problems += self.collect_logs()
        return problems

    def collect_logs(self) -> list[str]:
        """Copy error logs written by DataLab to the evidence folder."""
        problems = []
        candidates = [osp.join(self.cwd, "datalab_error.log")]
        for root, _dirs, names in os.walk(self.profile):
            candidates += [osp.join(root, n) for n in names if APP_LOG_RE.search(n)]
        for path in candidates:
            if not osp.isfile(path) or osp.getsize(path) == 0:
                continue
            name = osp.basename(path).lstrip(".")
            shutil.copy(path, osp.join(self.evidence, f"{self.name}-{name}"))
            problems.append(f"{name} written (copied to the evidence folder)")
            with open(path, encoding="utf-8", errors="replace") as stream:
                log(f"----- {path}\n{stream.read()}\n-----")
        return problems


def check_gui(
    exe: str, workdir: str, evidence: str, delay: int, timeout: float
) -> Callable[[], str]:
    """Return a step starting DataLab in unattended mode, with screenshots."""

    def step() -> str:
        shots = osp.join(evidence, "screenshots")
        run = ExecutableRun(exe, workdir, evidence, "gui")
        arguments = ["--unattended", "--screenshot", "--screenshot_path", shots]
        problems = run.execute(arguments + ["--delay", str(delay)], timeout)
        names = []
        if osp.isdir(shots):
            names = sorted(n for n in os.listdir(shots) if n.lower().endswith(".png"))
        if not names:
            problems.append("no screenshot was taken")
        if problems:
            raise SmokeTestError("; ".join(problems))
        return f"main window started and closed, screenshots: {', '.join(names)}"

    return step


def check_installation(
    exe: str, workdir: str, evidence: str, timeout: float
) -> Callable[[], str]:
    """Return a step running ``DataLab.exe --smoke-test``."""

    def step() -> str:
        report_path = osp.join(evidence, "smoke_test.json")
        run = ExecutableRun(exe, workdir, evidence, "probe")
        problems = run.execute(["--smoke-test", report_path], timeout)
        if not osp.isfile(report_path):
            raise SmokeTestError("; ".join(problems + ["no report was written"]))
        with open(report_path, encoding="utf-8") as stream:
            report = json.load(stream)
        if not report["frozen"]:
            problems.append("the executable is not the standalone version")
        for check in report["checks"]:
            log(f"[{check['status']}] {check['name']}: {check['detail']}")
            if check["status"] == "failed":
                log(check.get("traceback", ""))
                problems.append(f"{check['name']}: {check['detail']}")
            elif check["status"] != "passed":
                problems.append(f"{check['name']} {check['status']}")
        if problems:
            raise SmokeTestError("; ".join(problems))
        return f"{len(report['checks'])} checks passed"

    return step


class MsiSmokeTest:
    """Install, check and uninstall the MSI (``all`` command)."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.msi = osp.abspath(args.msi)
        self.dist = osp.abspath(args.dist)
        self.artifacts = [self.msi] + [osp.abspath(path) for path in args.artifact]
        self.workdir = osp.abspath(args.workdir)
        self.installdir = osp.join(self.workdir, "install")
        self.per_machine = args.per_machine
        self.delay = args.delay
        self.timeout = args.timeout
        version = args.version
        if version is None:
            match = re.fullmatch(r"DataLab-(.+)\.msi", osp.basename(self.msi))
            if match is None:
                raise SmokeTestError("Cannot read the version in the MSI file name")
            version = match.group(1)
        self.version = version
        guid = load_makewxs().get_product_code(version).upper()
        self.product_code = "{" + guid + "}"
        self.hashes: dict[str, str] = {}
        self.installed_files: dict[str, int] = {}
        self.installed = False

    def check_not_installed(self) -> str:
        """Refuse to run if DataLab of the same major version is installed."""
        state = get_product_state(self.product_code)
        if state != INSTALLSTATE_UNKNOWN:
            raise SmokeTestError(
                f"DataLab {self.version.split('.', 1)[0]}.x is already installed "
                f"on this computer (ProductCode {self.product_code}, state {state}): "
                "it would be uninstalled by this test, uninstall it first"
            )
        return f"no installation with ProductCode {self.product_code}"

    def hash_artifacts(self) -> str:
        """Hash the files which must not be modified."""
        self.hashes = {path: sha256(path) for path in self.artifacts}
        return ", ".join(
            f"{osp.basename(path)}: {digest[:16]}"
            for path, digest in self.hashes.items()
        )

    def install(self, evidence: str) -> str:
        """Install the MSI silently in the work directory."""
        if self.per_machine:
            scope = "ALLUSERS=1"
        else:
            scope = "ALLUSERS=2 MSIINSTALLPERUSER=1"
        code = run_msiexec(
            f'/i "{self.msi}" {scope} INSTALLFOLDER="{self.installdir}"',
            osp.join(evidence, "msi-install.log"),
        )
        if code == MSI_OTHER_VERSION:
            raise SmokeTestError("another version of DataLab is installed")
        if code not in MSI_SUCCESS:
            raise SmokeTestError(f"msiexec exit code {code} (see msi-install.log)")
        self.installed = True
        state = get_product_state(self.product_code)
        if state != INSTALLSTATE_DEFAULT:
            raise SmokeTestError(
                f"installed product state is {state} for ProductCode "
                f"{self.product_code}: the ProductCode computed by wix/makewxs.py "
                "does not match the MSI"
            )
        scope = "per machine" if self.per_machine else "per user"
        return f"installed in {self.installdir} ({scope})"

    def compare_files(self) -> str:
        """Compare the installed files with the PyInstaller folder."""
        expected = list_files(self.dist)
        installed = list_files(self.installdir)
        missing = sorted(set(expected) - set(installed))
        different = sorted(
            path
            for path, size in expected.items()
            if path in installed and installed[path] != size
        )
        self.installed_files = installed
        if missing or different:
            raise SmokeTestError(
                f"{len(missing)} missing files {missing[:10]}, "
                f"{len(different)} files of different size {different[:10]}"
            )
        return f"{len(expected)} files installed, {len(installed)} files in total"

    def uninstall(self, evidence: str, steps: Steps) -> str:
        """Uninstall the MSI and check that its files were removed."""
        code = run_msiexec(
            f"/x {self.product_code}", osp.join(evidence, "msi-uninstall.log")
        )
        if code not in MSI_SUCCESS:
            raise SmokeTestError(f"msiexec exit code {code} (see msi-uninstall.log)")
        self.installed = False
        remaining = list_files(self.installdir) if osp.isdir(self.installdir) else {}
        not_removed = sorted(set(remaining) & set(self.installed_files))
        if not_removed:
            raise SmokeTestError(
                f"{len(not_removed)} installed files were not removed: "
                f"{not_removed[:10]}"
            )
        if remaining:
            steps.warn(
                f"{len(remaining)} files created after installation remain after "
                f"uninstallation: {sorted(remaining)[:10]}"
            )
        shutil.rmtree(self.installdir, ignore_errors=True)
        return "uninstalled"

    def check_hashes(self) -> str:
        """Check that the hashed files were not modified."""
        modified = [
            path for path, digest in self.hashes.items() if sha256(path) != digest
        ]
        if modified:
            raise SmokeTestError(f"modified files: {modified}")
        return f"{len(self.hashes)} files unchanged"

    def run(self) -> int:
        """Run all steps, uninstalling the MSI even if a step failed."""
        self.check_not_installed()
        evidence = prepare_workdir(self.workdir)
        steps = Steps(f"Smoke test of {osp.basename(self.msi)}", evidence)
        exe = osp.join(self.installdir, "DataLab.exe")
        try:
            steps.run("Check that DataLab is not installed", self.check_not_installed)
            steps.run("Hash release artifacts", self.hash_artifacts)
            steps.run("Install the MSI", lambda: self.install(evidence))
            steps.run("Compare installed files", self.compare_files)
            steps.run(
                "Start the installed DataLab",
                check_gui(exe, self.workdir, evidence, self.delay, self.timeout),
            )
            steps.run(
                "Check the installation (--smoke-test)",
                check_installation(exe, self.workdir, evidence, self.timeout),
            )
        except SmokeTestError:
            pass
        finally:
            if self.installed:
                try:
                    steps.run(
                        "Uninstall the MSI", lambda: self.uninstall(evidence, steps)
                    )
                except SmokeTestError:
                    pass
            if self.hashes:
                try:
                    steps.run(
                        "Check that artifacts were not modified", self.check_hashes
                    )
                except SmokeTestError:
                    pass
        return steps.finish(msi=self.msi, version=self.version)


def run_executable_step(args: argparse.Namespace) -> int:
    """Run the ``gui`` or ``probe`` command on an executable."""
    exe = osp.abspath(args.exe)
    workdir = osp.abspath(args.workdir)
    evidence = prepare_workdir(workdir)
    steps = Steps(f"Smoke test of {exe}", evidence)
    if args.command == "gui":
        name, step = (
            "Start DataLab",
            check_gui(exe, workdir, evidence, args.delay, args.timeout),
        )
    else:
        name = "Check the installation (--smoke-test)"
        step = check_installation(exe, workdir, evidence, args.timeout)
    try:
        steps.run(name, step)
    except SmokeTestError:
        pass
    return steps.finish(executable=exe)


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run the smoke test."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        "all": subparsers.add_parser(
            "all", help="install, check and uninstall the MSI"
        ),
        "gui": subparsers.add_parser(
            "gui", help="start an executable with screenshots"
        ),
        "probe": subparsers.add_parser("probe", help="run an executable --smoke-test"),
    }
    for command, sub in commands.items():
        sub.add_argument(
            "--workdir", required=True, help="work directory (created, emptied)"
        )
        sub.add_argument(
            "--timeout", type=float, default=300.0, help="timeout of each run (s)"
        )
        if command == "all":
            sub.add_argument("--msi", required=True, help="MSI installer")
            sub.add_argument(
                "--dist", required=True, help="PyInstaller folder (dist/DataLab)"
            )
            sub.add_argument(
                "--artifact",
                action="append",
                default=[],
                help="other release artifact which must not be modified (repeatable)",
            )
            sub.add_argument(
                "--version", help="DataLab version (default: from the MSI file name)"
            )
            sub.add_argument(
                "--per-machine",
                action="store_true",
                help="install for all users (requires elevation)",
            )
        else:
            sub.add_argument("--exe", required=True, help="DataLab executable")
        if command != "probe":
            sub.add_argument(
                "--delay",
                type=int,
                default=8000,
                help="delay (ms) before DataLab quits (default: 8000)",
            )
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("this script runs on Windows only")
    try:
        if args.command == "all":
            return MsiSmokeTest(args).run()
        return run_executable_step(args)
    except SmokeTestError as exc:
        annotate("error", str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
