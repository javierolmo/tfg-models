#!/usr/bin/env python3
"""Unified entry point for tfg-models execution."""

import sys
from pathlib import Path

# Ensure src/ is on Python path if running uninstalled
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tfg_models.cli import main

if __name__ == "__main__":
    main()
