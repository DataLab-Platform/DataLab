@echo off
REM This script was copied from PythonQwt project
REM ======================================================
REM Documentation build script
REM ======================================================
REM Licensed under the terms of the MIT License
REM Copyright (c) 2020 Pierre Raybaut
REM (see PythonQwt LICENSE file for more details)
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion DATALAB_VERSION
cd %SCRIPTPATH%\..

@REM Clean previous documentation ======================================================
if exist %MODNAME%\data\doc ( rmdir /s /q %MODNAME%\data\doc )
mkdir %MODNAME%\data\doc

@REM Build documentation ===============================================================
@REM Screenshots under doc/images/ are NOT regenerated here: they are a maintainer
@REM responsibility (run scripts\update_screenshots.bat or the dedicated VS Code
@REM task "??? Refresh doc screenshots") and are committed as-is, which lets the
@REM CI doc workflows build the PDF without launching DataLab/Qt.
setlocal enabledelayedexpansion
for %%L in (fr en) do (
    set LANG=%%L
    if exist build\doc ( rmdir /s /q build\doc )
    sphinx-build -b latex -D language=%%L doc build\doc
    cd build\doc
    @REM Sphinx >= 9 emits a lowercased .tex filename (``datalab.tex``)
    @REM regardless of ``project = "DataLab"``. Auto-discover it instead
    @REM of hardcoding ``DataLab.tex``: NTFS hides the breakage locally
    @REM (case-insensitive lookup) but a case-sensitive filesystem fails.
    for %%F in (*.tex) do set "MAIN_TEX=%%F"
    echo Building PDF documentation for %%L from !MAIN_TEX!...
    @REM -enable-installer: let MiKTeX silently auto-install missing TeX
    @REM packages (e.g. ``noto`` and ``noto-emoji`` for the Unicode
    @REM fallback fonts referenced by doc/conf.py) instead of popping up
    @REM its Qt-based installer dialog (which fails when the venv shadows
    @REM Qt plugins). No effect on CI (Ubuntu + TeX Live).
    xelatex -enable-installer -interaction=nonstopmode -halt-on-error !MAIN_TEX!
    @REM Build again to fix table of contents (workaround)
    xelatex -enable-installer -interaction=nonstopmode -halt-on-error !MAIN_TEX!
    echo Done.
    cd ..\..
    move /Y build\doc\!MAIN_TEX:.tex=.pdf! %MODNAME%\data\doc\DataLab_%%L.pdf
)
endlocal

@REM explorer %MODNAME%\data\doc

call %FUNC% EndOfScript