"""Base model trainer providing a standardized training, evaluation, and inference template."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import settings
from core.model_handler import AzureModelHandler, LocalModelHandler, ModelHandler
from data.providers import AzureDataProvider, DataProvider, LocalDataProvider

logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """
    Template Method pattern base class for machine learning property valuation models.
    Standardizes data loading, preprocessing, model training, evaluation, and persistence.
    """

    def __init__(
        self,
        model_name: str,
        data_provider: Optional[DataProvider] = None,
        model_handler: Optional[ModelHandler] = None,
        test_size: float = settings.default_test_size,
        random_state: int = settings.default_random_state,
    ):
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self._data_provider = data_provider

        # Resolve model handler
        if model_handler:
            self.model_handler = model_handler
        elif settings.default_model_handler == "azure":
            self.model_handler = AzureModelHandler(model_name)
        else:
            self.model_handler = LocalModelHandler(model_name)

    @property
    def data_provider(self) -> DataProvider:
        """Lazily initialize DataProvider only when required (avoids starting Spark during inference)."""
        if self._data_provider is None:
            if settings.default_data_provider == "azure":
                self._data_provider = AzureDataProvider()
            else:
                self._data_provider = LocalDataProvider()
        return self._data_provider

    @abstractmethod
    def build_model(self) -> Any:
        """Initializes and returns the specific scikit-learn or lightgbm estimator."""
        pass

    @abstractmethod
    def preprocess(self, raw_spark_df: Any) -> Tuple[pd.DataFrame, pd.Series]:
        """Cleans and transforms raw data into features (X) and target (y)."""
        pass

    def evaluate(
        self,
        model: Any,
        features_test: pd.DataFrame,
        target_test: pd.Series,
    ) -> Dict[str, Any]:
        """Evaluates model performance and computes standard regression metrics."""
        logger.info("Evaluating model '%s'...", self.model_name)
        y_pred = model.predict(features_test)

        mae = mean_absolute_error(target_test, y_pred)
        mse = mean_squared_error(target_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(target_test, y_pred)

        report = {
            "model_name": self.model_name,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2_score": float(r2),
            "test_samples": len(target_test),
        }
        logger.info("Metrics for %s: MAE=%.2f, RMSE=%.2f, R2=%.4f", self.model_name, mae, rmse, r2)
        return report

    def run(self, save: bool = True) -> Dict[str, Any]:
        """Executes the full training and evaluation lifecycle."""
        logger.info("=== Starting training run for model: %s ===", self.model_name)

        # 1. Load Data
        logger.info("Loading properties dataset...")
        raw_spark_df = self.data_provider.read_properties_full()

        # 2. Preprocess
        features, target = self.preprocess(raw_spark_df)
        total_samples = len(target)
        logger.info("Data prepared with %d samples and %d features.", total_samples, features.shape[1])

        # 3. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # 4. Train Model
        logger.info("Training %s estimator...", self.model_name)
        model = self.build_model()
        model.fit(X_train, y_train)

        # 5. Evaluate
        report = self.evaluate(model, X_test, y_test)
        report["total_samples"] = total_samples
        report["train_samples"] = len(y_train)

        # 6. Save Model & Metadata
        if save:
            version_id = self.model_handler.save_model(
                model=model,
                features=list(X_train.columns),
                report=report,
            )
            report["saved_version"] = version_id
            logger.info("Model saved under version: %s", version_id)

        logger.info("=== Completed training run for model: %s ===", self.model_name)
        return report

    def predict_sample(
        self,
        surface: int,
        rooms: int,
        bathrooms: int,
        postal_code: int,
        elevator: bool = False,
        terrace: bool = False,
        garage: bool = False,
        version: str = "latest",
    ) -> float:
        """Helper to run single property valuation inference."""
        model, features = self.model_handler.load_model(version=version)
        sample = {
            "surface": [surface],
            "rooms": [rooms],
            "bathrooms": [bathrooms],
            "elevator": [int(elevator)],
            "terrace": [int(terrace)],
            "garage": [int(garage)],
        }
        return self._format_and_predict(model, features, sample, postal_code)

    @abstractmethod
    def _format_and_predict(
        self,
        model: Any,
        features: list,
        sample: dict,
        postal_code: int,
    ) -> float:
        """Subclass implementation of input formatting and prediction."""
        pass
