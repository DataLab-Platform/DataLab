
@echo off
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion CDL_VERSION

set BUILDDIR=%SCRIPTPATH%\..\build\gettext
cd %SCRIPTPATH%\..\doc
sphinx-build . %BUILDDIR% -b gettext
sphinx-intl update -p %BUILDDIR% -l fr
call %~dp0gettext rescan