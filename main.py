"""Root entry point for tfg-models CLI."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cli import main

if __name__ == "__main__":
    main()
