"""Backwards-compatibility wrapper for model handler."""

import sys
from pathlib import Path

src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from tfg_models.core.model_handler import AzureModelHandler, ModelHandler

__all__ = ["ModelHandler", "AzureModelHandler"]
