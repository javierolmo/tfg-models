"""LightGBM Regressor model trainer implementation."""

import logging
from typing import Any, Dict

import lightgbm as lgb
import pandas as pd

from tfg_models.core.trainer import BaseModelTrainer

logger = logging.getLogger(__name__)


class LightGBMTrainer(BaseModelTrainer):
    """LightGBM model trainer using native categorical support for postal codes."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        **kwargs,
    ):
        super().__init__(
            model_name="lightgbm",
            encoding="categorical",
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves

    def fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Trains a LightGBM Regressor with native categorical features."""
        logger.info(
            "Fitting LightGBM Regressor (n_estimators=%d, lr=%.3f, num_leaves=%d)...",
            self.n_estimators,
            self.learning_rate,
            self.num_leaves,
        )
        model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            categorical_feature=["postal_code"],
        )
        return model

    def evaluate(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates LightGBM and appends top feature importances to report."""
        report = super().evaluate(model, X_test, y_test)

        feature_names = model.feature_name_
        importances = model.feature_importances_
        importance_dict = dict(
            sorted(
                zip(feature_names, [int(imp) for imp in importances]),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        report["feature_importances"] = importance_dict
        logger.info("[%s] Top feature importances: %s", self.model_name, list(importance_dict.items())[:5])
        return report
