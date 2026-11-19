# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller Specification Manifest for Aurelix Vision Official Standalone Executable."""

block_cipher = None

a = Analysis(
    ['aurelix.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('configs', 'configs'),
        ('datasets', 'datasets'),
        ('aurelix', 'aurelix'),
    ],
    hiddenimports=[
        'aurelix',
        'aurelix.launcher',
        'aurelix.launcher.app',
        'aurelix.launcher.hardware',
        'aurelix.launcher.managers',
        'aurelix.launcher.ui',
        'aurelix.models',
        'aurelix.losses',
        'aurelix.trainer',
        'aurelix.datasets',
        'aurelix.export',
        'aurelix.metrics',
        'aurelix.callbacks',
        'aurelix.utils',
        'aurelix.deployment',
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
    name='AurelixVision',
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
