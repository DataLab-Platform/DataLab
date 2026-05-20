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
for %%L in (fr en) do (
    set LANG=%%L
    if exist build\doc ( rmdir /s /q build\doc )
    sphinx-build -b latex -D language=%%L doc build\doc
    cd build\doc
    echo Building PDF documentation for %%L...
    xelatex -interaction=nonstopmode -quiet DataLab.tex
    @REM Build again to fix table of contents (workaround)
    xelatex -interaction=nonstopmode -quiet DataLab.tex
    echo Done.
    cd ..\..
    move /Y build\doc\DataLab.pdf %MODNAME%\data\doc\DataLab_%%L.pdf
)

@REM explorer %MODNAME%\data\doc

call %FUNC% EndOfScript