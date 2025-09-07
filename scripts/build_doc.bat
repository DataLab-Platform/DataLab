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

@REM Set light mode for Qt applications and clean previous documentation ===============
set QT_COLOR_MODE=light
if exist %MODNAME%\data\doc ( rmdir /s /q %MODNAME%\data\doc )
mkdir %MODNAME%\data\doc

@REM Build documentation ===============================================================
for %%L in (fr en) do (
    @REM -------------------------------------------------------------------------------
    @REM Create dummy PDF file, otherwise the PDF menu entry in "?" menu
    @REM won't be visible in the automatic screenshot
    echo Dummy PDF file > %MODNAME%\data\doc\DataLab_%%L.pdf
    @REM -------------------------------------------------------------------------------
    set LANG=%%L
    %PYTHON% doc/update_screenshots.py
    if exist build\doc ( rmdir /s /q build\doc )
    sphinx-build -b latex -D language=%%L doc build\doc
    cd build\doc
    echo Building PDF documentation for %%L...
    pdflatex -interaction=nonstopmode -quiet DataLab.tex
    @REM Build again to fix table of contents (workaround)
    pdflatex -interaction=nonstopmode -quiet DataLab.tex
    echo Done.
    cd ..\..
    move /Y build\doc\DataLab.pdf %MODNAME%\data\doc\DataLab_%%L.pdf
)

@REM explorer %MODNAME%\data\doc

call %FUNC% EndOfScript