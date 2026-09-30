"""Backwards-compatibility wrapper for model handlers."""

from core.model_handler import AzureModelHandler, LocalModelHandler, ModelHandler

__all__ = ["ModelHandler", "LocalModelHandler", "AzureModelHandler"]
