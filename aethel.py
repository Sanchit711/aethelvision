#!/usr/bin/env python
"""AETHEL Enterprise AI Vision Platform — Master Console Launcher.

Official application entry point for AETHEL Enterprise AI Vision Platform.
Orchestrates hardware inspection, dataset management, model training, benchmarking,
evaluation, export, and deployment with 100% backend compatibility.

Usage:
    python aethel.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from aethel.launcher.app import AethelApplicationLauncher


def main() -> None:
    """Launch official AETHEL Enterprise AI Vision Platform console."""
    app = AethelApplicationLauncher(root_dir=root_dir)
    app.run()


if __name__ == "__main__":
    main()
