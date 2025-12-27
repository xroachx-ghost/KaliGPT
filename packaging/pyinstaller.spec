# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

block_cipher = None

analysis = Analysis(
    [str(project_root / "src" / "main.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[(str(project_root / "packaging" / "version.json"), "packaging")],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    [],
    name="KaliGPT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
