# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

import os.path as osp
import guidata
guidata_path = osp.dirname(guidata.__file__)
guidata_images = osp.join(guidata_path, 'images')
guidata_locale = osp.join(guidata_path, 'locale', 'fr', 'LC_MESSAGES')
import plotpy
plotpy_path = osp.dirname(plotpy.__file__)
plotpy_images = osp.join(plotpy_path, 'images')
plotpy_locale = osp.join(plotpy_path, 'locale', 'fr', 'LC_MESSAGES')

from PyInstaller.utils.hooks import collect_submodules
all_hidden_imports = collect_submodules('cdl')

a = Analysis(['cdl\\start.pyw'],
             pathex=[],
             binaries=[],
             datas=[
                    (guidata_images, 'guidata\\images'),
                    (guidata_locale, 'guidata\\locale\\fr\\LC_MESSAGES'),
                    (plotpy_images, 'plotpy\\images'),
                    (plotpy_locale, 'plotpy\\locale\\fr\\LC_MESSAGES'),
                    ('cdl\\plugins', 'cdl\\plugins'),
                    ('cdl\\data', 'cdl\\data'),
                    ('cdl\\locale\\fr\\LC_MESSAGES\\cdl.mo', 'cdl\\locale\\fr\\LC_MESSAGES'),
                    ],
             hiddenimports=all_hidden_imports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
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
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None , icon='resources\\DataLab.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='DataLab')
