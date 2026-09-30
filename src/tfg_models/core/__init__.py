"""Core package for tfg-models."""

from tfg_models.core.model_handler import AzureModelHandler, LocalModelHandler, ModelHandler
from tfg_models.core.trainer import BaseModelTrainer

__all__ = [
    "ModelHandler",
    "LocalModelHandler",
    "AzureModelHandler",
    "BaseModelTrainer",
]
