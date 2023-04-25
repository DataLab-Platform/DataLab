set INKSCAPE_PATH="C:\Program Files\Inkscape\bin\inkscape.exe"

@REM Deploying images and icons
copy DataLab.svg ..\cdl\data\logo
%INKSCAPE_PATH% "DataLab-Title.svg" -o "DataLab-Title.png" -w 190
copy DataLab-Title.png ..\doc\_static
%INKSCAPE_PATH% "DataLab-Splash.svg" -o "DataLab-Splash.png" -w 300
copy DataLab-Splash.png ..\cdl\data\logo
%INKSCAPE_PATH% "DataLab-Watermark.svg" -o "..\cdl\data\logo\DataLab-Watermark.png" -w 300
%INKSCAPE_PATH% "DataLab-Banner.svg" -o "DataLab-Banner.png" -w 364
copy DataLab-Banner.png ..\doc\images

@REM Generating icon
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "DataLab.svg" -o "tmp-%%s.png" -w %%s -h %%s
)
magick convert "tmp-*.png" "DataLab.ico"
del "tmp-*.png"

@REM Generating images for NSIS installer
%INKSCAPE_PATH% "win.svg" -o "temp.png" -w 164 -h 314
magick convert "temp.png" "win.bmp"
%INKSCAPE_PATH% "banner.svg" -o "temp.png" -w 300 -h 114
magick convert "temp.png" "banner.bmp"
del "temp.png"
move /y *.bmp ..\nsis\images

@REM Generating icons for NSIS installer
for %%s in (16 24 32 48 128 256) do (
  %INKSCAPE_PATH% "install.svg" -o "install-%%s.png" -w %%s -h %%s
  %INKSCAPE_PATH% "uninstall.svg" -o "uninstall-%%s.png" -w %%s -h %%s
)
magick convert "install-*.png" "install.ico"
magick convert "uninstall-*.png" "uninstall.ico"
del "install-*.png"
del "uninstall-*.png"
move /y *install.ico ..\nsis\icons
