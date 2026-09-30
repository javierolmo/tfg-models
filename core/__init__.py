"""Core package for tfg-models."""

from core.model_handler import AzureModelHandler, LocalModelHandler, ModelHandler
from core.trainer import BaseModelTrainer

__all__ = [
    "ModelHandler",
    "LocalModelHandler",
    "AzureModelHandler",
    "BaseModelTrainer",
]
