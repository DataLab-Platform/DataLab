@echo off

call %~dp0utils GetScriptPath SCRIPTPATH
call %FUNC% GetLibName LIBNAME
call %FUNC% GetModName MODNAME
call %FUNC% SetPythonPath
call %FUNC% UsePython
call %FUNC% GetVersion VERSION

echo ===========================================================================
echo Making DataLab v%VERSION% release with %WINPYDIRBASE%
echo ===========================================================================

set destdir=releases\DataLab-v%VERSION%-release
if exist %destdir% ( rmdir /s /q %destdir% )
mkdir %destdir%
move "dist\*.whl" %destdir%
move "dist\*.gz" %destdir%
move "dist\*.zip" %destdir%
move "wix\DataLab-%VERSION%.msi" %destdir%
copy "CHANGELOG.md" %destdir%
copy %MODNAME%\data\doc\*.pdf %destdir%
explorer %destdir%

call %FUNC% EndOfScript