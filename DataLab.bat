@echo off
REM Check if a Python executable path is provided as an argument
if "%1" == "" (
    echo Error: Python executable path must be provided as an argument.
    echo Usage: DataLab.bat path\to\python.exe
    exit /b 1
) else (
    REM Use the provided Python executable path
    set PYTHON_EXE=%1
)
REM Validate that the provided Python executable exists
if not exist %PYTHON_EXE% (
    echo Error: The specified Python executable does not exist: %PYTHON_EXE%
    exit /b 2
)

cd/D %~dp0
set ORIGINAL_PYTHONPATH=%PYTHONPATH%
for /F "tokens=*" %%A in (.env) do (set %%A)
set PYTHONPATH=%PYTHONPATH%;%ORIGINAL_PYTHONPATH%

REM Extract pythonw.exe from the same directory as the provided python.exe
for %%a in ("%PYTHON_EXE%") do set "PYTHON_DIR=%%~dpa"
start "" "%PYTHON_DIR%pythonw.exe" datalab\start.pyw %2 %3 %4 %5 %6 %7 %8 %9