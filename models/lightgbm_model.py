"""LightGBM gradient boosting model trainer implementation."""

import logging
from typing import Any, Dict, Tuple

import lightgbm as lgb
import pandas as pd

from core.trainer import BaseModelTrainer
from data.preprocessing import clean_property_data, prepare_features_for_model

logger = logging.getLogger(__name__)


class LightGBMTrainer(BaseModelTrainer):
    """Trainer for LightGBM Regressor with native categorical handling for postal codes."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        **kwargs,
    ):
        super().__init__(model_name="lightgbm", **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def build_model(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            objective="regression",
            verbosity=-1,
        )

    def preprocess(self, raw_spark_df: Any) -> Tuple[pd.DataFrame, pd.Series]:
        cleaned_df = clean_property_data(raw_spark_df)
        return prepare_features_for_model(cleaned_df, encoding="categorical")

    def evaluate(
        self,
        model: lgb.LGBMRegressor,
        features_test: pd.DataFrame,
        target_test: pd.Series,
    ) -> Dict[str, Any]:
        report = super().evaluate(model, features_test, target_test)

        importances = model.feature_importances_
        feature_names = list(features_test.columns)
        importance_dict = {
            name: int(imp) for name, imp in zip(feature_names, importances)
        }
        report["feature_importances"] = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        )
        report["hyperparameters"] = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
        }
        return report

    def _format_and_predict(
        self,
        model: Any,
        features: list,
        sample: dict,
        postal_code: int,
    ) -> float:
        sample["postal_code"] = [postal_code]
        df = pd.DataFrame(sample)
        df["postal_code"] = df["postal_code"].astype("category")
        df = df[features]
        prediction = float(model.predict(df)[0])
        return prediction
