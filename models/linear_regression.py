"""Linear regression model trainer implementation."""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.trainer import BaseModelTrainer
from data.preprocessing import clean_property_data, prepare_features_for_model

logger = logging.getLogger(__name__)


class LinearRegressionTrainer(BaseModelTrainer):
    """Trainer for Ordinary Least Squares (OLS) Linear Regression with one-hot encoded postal codes."""

    def __init__(self, **kwargs):
        super().__init__(model_name="linear_regression", **kwargs)

    def build_model(self) -> LinearRegression:
        return LinearRegression()

    def preprocess(self, raw_spark_df: Any) -> Tuple[pd.DataFrame, pd.Series]:
        cleaned_df = clean_property_data(raw_spark_df)
        return prepare_features_for_model(cleaned_df, encoding="onehot")

    def evaluate(
        self,
        model: LinearRegression,
        features_test: pd.DataFrame,
        target_test: pd.Series,
    ) -> Dict[str, Any]:
        report = super().evaluate(model, features_test, target_test)
        report["intercept"] = float(model.intercept_)
        return report

    def _format_and_predict(
        self,
        model: Any,
        features: list,
        sample: dict,
        postal_code: int,
    ) -> float:
        sample[f"postal_code_{postal_code}"] = [1]
        df = pd.DataFrame(sample)
        df = df.reindex(columns=features, fill_value=0)
        prediction = float(model.predict(df)[0])
        return prediction
