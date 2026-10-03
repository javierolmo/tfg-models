"""TFG Real Estate Valuation Models Package."""

# Pre-initialize PyTorch runtime before C-extensions/OpenMP runtimes
try:
    import torch  # noqa: F401
except ImportError:
    pass

__version__ = "1.1.0"

from tfg_models.config import settings
from tfg_models.core.model_handler import (
    AzureModelHandler,
    LocalModelHandler,
    ModelHandler,
    get_model_handler,
)
from tfg_models.core.trainer import BaseModelTrainer
from tfg_models.data.providers import (
    AzureDataProvider,
    DataProvider,
    LocalDataProvider,
)
from tfg_models.models import (
    LightGBMTrainer,
    LinearRegressionTrainer,
    NeuralNetworkTrainer,
    get_model_trainer,
)

__all__ = [
    "__version__",
    "settings",
    "ModelHandler",
    "AzureModelHandler",
    "LocalModelHandler",
    "get_model_handler",
    "BaseModelTrainer",
    "DataProvider",
    "AzureDataProvider",
    "LocalDataProvider",
    "LinearRegressionTrainer",
    "LightGBMTrainer",
    "NeuralNetworkTrainer",
    "get_model_trainer",
]
