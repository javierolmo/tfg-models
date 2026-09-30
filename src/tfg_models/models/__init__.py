"""Models registry and factory for tfg-models."""

from typing import Dict, Type

from tfg_models.core.trainer import BaseModelTrainer
from tfg_models.models.lightgbm_model import LightGBMTrainer
from tfg_models.models.linear_regression import LinearRegressionTrainer

MODEL_REGISTRY: Dict[str, Type[BaseModelTrainer]] = {
    "linear_regression": LinearRegressionTrainer,
    "lightgbm": LightGBMTrainer,
}


def get_model_trainer(model_name: str, **kwargs) -> BaseModelTrainer:
    """Factory creating an instance of a registered model trainer."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)


__all__ = [
    "MODEL_REGISTRY",
    "get_model_trainer",
    "LinearRegressionTrainer",
    "LightGBMTrainer",
]
