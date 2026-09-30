"""TFG Real Estate Valuation Models Package."""

__version__ = "0.0.1"

from tfg_models.config import settings
from tfg_models.core.model_handler import (
    AzureModelHandler,
    LocalModelHandler,
    ModelHandler,
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
    get_model_trainer,
)

__all__ = [
    "__version__",
    "settings",
    "ModelHandler",
    "LocalModelHandler",
    "AzureModelHandler",
    "BaseModelTrainer",
    "DataProvider",
    "AzureDataProvider",
    "LocalDataProvider",
    "LinearRegressionTrainer",
    "LightGBMTrainer",
    "get_model_trainer",
]
