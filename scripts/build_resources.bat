@echo off
REM ======================================================
REM Graphics resources build script
REM ------------------------------------------------------
REM Regenerates committed binary resources from SVG sources:
REM   - resources\DataLab.ico  (multi-size icon for the EXE)
REM   - wix\dialog.bmp         (493x312 WiX UI dialog background)
REM   - wix\banner.bmp         (493x58 WiX UI banner)
REM
REM Run this script ONLY when the source SVG files change.
REM The generated files are committed under git so that the
REM release pipeline does not need any external rasteriser.
REM
REM Requirements: Qt (via qtpy/PyQt5) and Pillow, both already
REM pulled transitively by DataLab. No external tool required.
REM ======================================================
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% UsePython

%PYTHON% "%SCRIPTPATH%\build_resources.py" %*

call %FUNC% EndOfScript
