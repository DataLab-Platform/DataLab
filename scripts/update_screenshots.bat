@echo off
REM ======================================================
REM Refresh DataLab documentation screenshots
REM ======================================================
REM Maintainer-only task: regenerates the PNG files under
REM doc/images/ by launching DataLab via Xvfb-free Qt and
REM capturing the relevant dialogs / panels. The results
REM are committed assets (see CONTRIBUTING / doc).
REM
REM Re-run this after any change that affects the rendered
REM UI (menus, dialogs, panels, themes) and commit the
REM updated PNGs in a dedicated "docs: refresh screenshots"
REM commit.
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
cd %SCRIPTPATH%\..

set QT_COLOR_MODE=light

@REM Dummy PDFs so the "?" menu entry is visible in screenshots
if not exist %MODNAME%\data\doc ( mkdir %MODNAME%\data\doc )
for %%L in (fr en) do (
    if not exist %MODNAME%\data\doc\DataLab_%%L.pdf (
        echo Dummy PDF file > %MODNAME%\data\doc\DataLab_%%L.pdf
    )
)

for %%L in (fr en) do (
    set LANG=%%L
    %PYTHON% doc/update_screenshots.py
)

call %FUNC% EndOfScript
