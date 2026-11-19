"""Root conftest.py to ensure project root directory is in sys.path."""

import os
import sys

# Ensure root directory is at index 0 of sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
