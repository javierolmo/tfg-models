"""Abstract base trainer defining the Template Method pattern for ML models."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tfg_models.config import settings
from tfg_models.core.model_handler import AzureModelHandler, ModelHandler
from tfg_models.data.preprocessing import clean_property_data, prepare_features_for_model
from tfg_models.data.providers import AzureDataProvider, DataProvider, LocalDataProvider

logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """
    Template Method pattern base class for property valuation models.
    Orchestrates the entire lifecycle: load -> preprocess -> train -> evaluate -> persist.
    """

    def __init__(
        self,
        model_name: str,
        encoding: str = "onehot",
        test_size: float = 0.2,
        random_state: int = 42,
        data_provider: Optional[DataProvider] = None,
        model_handler: Optional[ModelHandler] = None,
    ):
        self.model_name = model_name
        self.encoding = encoding
        self.test_size = test_size
        self.random_state = random_state

        self._data_provider = data_provider
        self._model_handler = model_handler

    @property
    def data_provider(self) -> DataProvider:
        """Lazily initializes the data provider to avoid loading PySpark on inference."""
        if self._data_provider is None:
            if settings.default_data_provider == "azure":
                self._data_provider = AzureDataProvider()
            else:
                self._data_provider = LocalDataProvider()
        return self._data_provider

    @property
    def model_handler(self) -> ModelHandler:
        """Lazily initializes the model persistence handler (defaults to production AzureModelHandler)."""
        if self._model_handler is None:
            self._model_handler = AzureModelHandler(self.model_name)
        return self._model_handler

    def load_data(self) -> pd.DataFrame:
        """Loads properties_full via the data provider and returns a pandas DataFrame."""
        logger.info("[%s] Step 1: Loading raw data from data provider...", self.model_name)
        spark_df = self.data_provider.read_properties_full()
        pdf = clean_property_data(spark_df)
        logger.info("[%s] Cleaned dataset shape: %s", self.model_name, pdf.shape)
        return pdf

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepares features and target from cleaned data."""
        logger.info("[%s] Step 2: Preparing features (encoding: %s)...", self.model_name, self.encoding)
        X, y = prepare_features_for_model(df, encoding=self.encoding)
        return X, y, list(X.columns)

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits features into train and test partitions."""
        logger.info(
            "[%s] Step 3: Splitting train/test (test_size=%.2f, random_state=%d)...",
            self.model_name,
            self.test_size,
            self.random_state,
        )
        return train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

    @abstractmethod
    def fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Model-specific training implementation."""
        pass

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates model performance and returns metric summary."""
        logger.info("[%s] Step 5: Evaluating model performance...", self.model_name)
        predictions = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        report = {
            "model_name": self.model_name,
            "encoding": self.encoding,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "test_samples": len(y_test),
        }

        logger.info(
            "[%s] Evaluation results -> MAE: %.2f € | RMSE: %.2f € | R²: %.4f",
            self.model_name,
            mae,
            rmse,
            r2,
        )
        return report

    def save(
        self,
        model: Any,
        features: List[str],
        report: Dict[str, Any],
        version: Optional[str] = None,
    ) -> str:
        """Persists model and features using the configured handler."""
        logger.info("[%s] Step 6: Persisting model and artifacts...", self.model_name)
        return self.model_handler.save_model(model, features, report, version=version)

    def run(self, version: Optional[str] = None) -> Tuple[Any, List[str], Dict[str, Any], str]:
        """Template Method executing the full ML training lifecycle."""
        logger.info("========== Starting pipeline for '%s' ==========", self.model_name)
        df = self.load_data()
        X, y, features = self.prepare_features(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        logger.info("[%s] Step 4: Training model on %d samples...", self.model_name, len(X_train))
        model = self.fit_model(X_train, y_train)

        report = self.evaluate(model, X_test, y_test)
        version_id = self.save(model, features, report, version=version)
        logger.info("========== Pipeline finished for '%s' (version: %s) ==========", self.model_name, version_id)
        return model, features, report, version_id

    def predict_sample(
        self,
        surface: float,
        rooms: int,
        bathrooms: int,
        postal_code: str,
        elevator: bool = False,
        terrace: bool = False,
        garage: bool = False,
        version: str = "latest",
    ) -> float:
        """Loads persisted model and performs inference on a single property sample."""
        model, features = self.model_handler.load_model(version=version)

        input_data = pd.DataFrame(
            [
                {
                    "surface": surface,
                    "rooms": rooms,
                    "bathrooms": bathrooms,
                    "elevator": elevator,
                    "terrace": terrace,
                    "garage": garage,
                    "postal_code": str(postal_code),
                }
            ]
        )

        if self.encoding == "onehot":
            for col in features:
                if col not in input_data.columns:
                    input_data[col] = False
            target_pc_col = f"postal_code_{postal_code}"
            if target_pc_col in input_data.columns:
                input_data[target_pc_col] = True
            X_input = input_data[features]
        else:
            input_data["postal_code"] = input_data["postal_code"].astype("category")
            X_input = input_data[features]

        prediction = model.predict(X_input)[0]
        return float(prediction)
