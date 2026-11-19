# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller packaging specification for AethelVision Enterprise AI Vision Platform."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

datas = [
    ('configs', 'configs'),
    ('assets', 'assets'),
]

hiddenimports = [
    'aethel',
    'aethel.engine.hal',
    'aethel.inference.engine',
    'aethel.inference.visualizer',
    'aethel.export.universal',
    'aethel.models',
    'aethel.models.aethelnet',
    'aethel.models.pose',
    'aethel.models.segment',
    'aethel.models.obb',
    'aethel.losses',
    'aethel.launcher.app',
]

a = Analysis(
    ['aethel.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib.tests'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AethelVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/logo.ico' if os.path.exists('assets/logo.ico') else None,
)
