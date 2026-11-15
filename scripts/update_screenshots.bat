@echo off
setlocal
REM ======================================================
REM Refresh DataLab documentation screenshots
REM ======================================================
REM Maintainer-only task: regenerates the PNG files under
REM doc/images/ by launching DataLab via offscreen Qt and
REM capturing the relevant dialogs / panels. The results
REM are committed assets (see CONTRIBUTING / doc).
REM
REM Re-run this after any change that affects the rendered
REM UI (menus, dialogs, panels, themes) and commit the
REM updated PNGs in a dedicated "docs: refresh screenshots"
REM commit.
REM ======================================================
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_with_env.py python doc\update_screenshots.py %*
) else (
    python scripts\run_with_env.py python doc\update_screenshots.py %*
)
exit /b %errorlevel%
