"""Models registry and factory."""

from typing import Dict, Type

from core.trainer import BaseModelTrainer
from models.lightgbm_model import LightGBMTrainer
from models.linear_regression import LinearRegressionTrainer

MODEL_REGISTRY: Dict[str, Type[BaseModelTrainer]] = {
    "linear_regression": LinearRegressionTrainer,
    "linear": LinearRegressionTrainer,
    "lightgbm": LightGBMTrainer,
    "lgbm": LightGBMTrainer,
}


def get_model_trainer(model_name: str, **kwargs) -> BaseModelTrainer:
    """Factory function to instantiate a model trainer by name."""
    normalized_name = model_name.strip().lower()
    if normalized_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {list(MODEL_REGISTRY.keys())}"
        )
    trainer_cls = MODEL_REGISTRY[normalized_name]
    return trainer_cls(**kwargs)


__all__ = [
    "LinearRegressionTrainer",
    "LightGBMTrainer",
    "MODEL_REGISTRY",
    "get_model_trainer",
]
