# -*- mode: python ; coding: utf-8 -*-

# Initial command:
# pyinstaller -y --clean -n DataLab -i resources\DataLab.ico cdl\start.pyw

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
all_hidden_imports = collect_submodules('cdl')
datas = collect_data_files('cdl') + [('cdl\\plugins', 'cdl\\plugins')]
datas += collect_data_files('guidata') + collect_data_files('plotpy')

a = Analysis(
    ['cdl\\start.pyw'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=all_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DataLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['resources\\DataLab.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DataLab',
)
