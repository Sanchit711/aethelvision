#!/usr/bin/env python
"""AURELIX Enterprise AI Vision Platform — Application Launcher & Entry Point.

Official application entry point for AURELIX Enterprise AI Vision Platform.
Orchestrates hardware inspection, dataset management, model training, benchmarking,
evaluation, export, and deployment while maintaining 100% backend compatibility.

Usage:
    python aurelix.py  (or python aurelix.py)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from aurelix.launcher.app import AurelixApplicationLauncher


def main() -> None:
    """Launch official AURELIX Enterprise AI Vision Platform console."""
    app = AurelixApplicationLauncher(root_dir=root_dir)
    app.run()


if __name__ == "__main__":
    main()

