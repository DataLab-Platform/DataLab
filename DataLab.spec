# -*- mode: python ; coding: utf-8 -*-

# Initial command:
# pyinstaller -y --clean -n DataLab -i resources\DataLab.ico datalab\start.pyw

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata
all_hidden_imports = collect_submodules('datalab')
# Plugins installed from a file may import any Sigima module
all_hidden_imports += collect_submodules('sigima', filter=lambda name: '.tests' not in name)
all_hidden_imports += collect_submodules('pip')
datas = collect_data_files('datalab') + [('datalab\\plugins\\builtin', 'datalab\\plugins\\builtin')]
datas += collect_data_files('pip') + copy_metadata('pip')
# Macro templates are read with ``pkgutil.get_data`` and must exist on disk,
# not only inside the PYZ archive (``collect_data_files`` skips ``.py`` files):
datas += [('datalab\\gui\\macros_templates', 'datalab\\gui\\macros_templates')]
datas += collect_data_files('guidata') + collect_data_files('plotpy')
datas += collect_data_files('sigima') + collect_data_files('sigimax')
datas += copy_metadata('imageio')
# Installed plugin wheels declare their requirements on these distributions
datas += copy_metadata('datalab-platform', recursive=True)
# Provenance and capsules: DataLab-Capsule reads its schemas and RO-Crate
# context with ``importlib.resources``, and the capsule dialog imports some of
# its modules lazily. The registry server (``registry_app``, FastAPI) and the
# command line are not part of the application.
all_hidden_imports += collect_submodules(
    'datalab_capsule',
    filter=lambda name: name not in ('datalab_capsule.registry_app', 'datalab_capsule.cli'),
)
datas += collect_data_files('datalab_capsule')

a = Analysis(
    ['datalab\\start.pyw'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=all_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    # distlib's resource finder requires a standard file-system loader.
    module_collection_mode={'pip': 'py'},
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
