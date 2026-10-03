"""Models registry and factory for tfg-models."""

# Import PyTorch neural network before LightGBM to ensure clean OpenMP initialization
try:
    import torch  # noqa: F401
except ImportError:
    pass

from typing import Dict, Type

from tfg_models.core.trainer import BaseModelTrainer
from tfg_models.models.neural_network import NeuralNetworkTrainer
from tfg_models.models.linear_regression import LinearRegressionTrainer
from tfg_models.models.lightgbm_model import LightGBMTrainer

MODEL_REGISTRY: Dict[str, Type[BaseModelTrainer]] = {
    "linear_regression": LinearRegressionTrainer,
    "lightgbm": LightGBMTrainer,
    "neural_network": NeuralNetworkTrainer,
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
    "NeuralNetworkTrainer",
]
