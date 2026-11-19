"""Cython Native Binary Compilation Script for Aethel Vision Framework.

Compiles sensitive proprietary algorithms (continual learning, security watermarking, engine)
into high-performance C machine code binaries (.pyd on Windows, .so on Linux/Mac):

Usage:
    python setup_cython.py build_ext --inplace
"""

import sys
from pathlib import Path
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
    has_cython = True
except ImportError:
    has_cython = False

MODULES_TO_COMPILE = [
    "aethel/continual/distillation.py",
    "aethel/continual/class_expansion.py",
    "aethel/continual/replay_buffer.py",
    "aethel/security/watermark.py",
]

ext_modules = []
if has_cython:
    for mod_path in MODULES_TO_COMPILE:
        if Path(mod_path).exists():
            mod_name = mod_path.replace("/", ".").replace("\\", ".").replace(".py", "")
            ext_modules.append(
                Extension(
                    name=mod_name,
                    sources=[mod_path],
                    extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3", "-march=native"],
                )
            )

if __name__ == "__main__":
    if not has_cython:
        print("[!] Cython is not installed. Run: pip install cython")
        print("[*] To compile .pyd on Windows, ensure Visual C++ Build Tools or MinGW is installed.")
        sys.exit(1)

    setup(
        name="aethel-native-binaries",
        ext_modules=cythonize(
            ext_modules,
            compiler_directives={
                "language_level": "3",
                "always_allow_keywords": True,
                "binding": True,
            },
        ),
    )
