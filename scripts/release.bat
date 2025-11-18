@echo off

call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION

echo ===========================================================================
echo Making DataLab v%VERSION% release with %PYTHON%
echo ===========================================================================

set destdir=releases\DataLab-v%VERSION%-release
if exist %destdir% ( rmdir /s /q %destdir% )
mkdir %destdir%
move "dist\*.whl" %destdir%
move "dist\*.gz" %destdir%
move "dist\*.zip" %destdir%
move "wix\DataLab-%VERSION%.msi" %destdir%
copy "CHANGELOG.md" %destdir%
move %MODNAME%\data\doc\*.pdf %destdir%
explorer %destdir% || exit /b 0

call %FUNC% EndOfScript