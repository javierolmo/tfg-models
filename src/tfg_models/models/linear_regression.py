"""Linear Regression model trainer implementation."""

import logging
from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression

from tfg_models.core.trainer import BaseModelTrainer

logger = logging.getLogger(__name__)


class LinearRegressionTrainer(BaseModelTrainer):
    """Linear regression model trainer using one-hot encoded postal codes."""

    def __init__(self, **kwargs):
        super().__init__(
            model_name="linear_regression",
            encoding="onehot",
            **kwargs,
        )

    def fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Trains a Scikit-Learn LinearRegression model."""
        logger.info("Fitting Scikit-Learn LinearRegression on %d samples...", len(X_train))
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
