"""Backwards-compatibility wrapper for data providers."""

import sys
from pathlib import Path

src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tfg_models.data.providers import AzureDataProvider, DataProvider, LocalDataProvider

__all__ = ["DataProvider", "AzureDataProvider", "LocalDataProvider"]
