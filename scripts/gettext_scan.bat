
@echo off
call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UseWinPython

@REM CODRAFT_VERSION is used by doc/conf.py to set documentation version
call %FUNC% GetVersion CODRAFT_VERSION

cd %SCRIPTPATH%\..\doc
sphinx-build . locale\pot -b gettext
sphinx-intl update -p locale\pot -l fr
call %~dp0gettext rescan