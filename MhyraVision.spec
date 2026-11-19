# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller Specification Manifest for Mhyra Vision Official Standalone Executable."""

block_cipher = None

a = Analysis(
    ['mhyra.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('configs', 'configs'),
        ('datasets', 'datasets'),
        ('mhyra', 'mhyra'),
    ],
    hiddenimports=[
        'mhyra',
        'mhyra.launcher',
        'mhyra.launcher.app',
        'mhyra.launcher.hardware',
        'mhyra.launcher.managers',
        'mhyra.launcher.ui',
        'mhyra.models',
        'mhyra.losses',
        'mhyra.trainer',
        'mhyra.datasets',
        'mhyra.export',
        'mhyra.metrics',
        'mhyra.callbacks',
        'mhyra.utils',
        'mhyra.deployment',
        'datasets.dataloader',
        'datasets.factory',
        'datasets.yolo',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='MhyraVision',
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
)
