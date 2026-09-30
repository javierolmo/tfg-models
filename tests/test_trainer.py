"""Unit tests for BaseModelTrainer template method and execution lifecycle."""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from tfg_models.core.trainer import BaseModelTrainer
from tests.helpers.local_model_handler import LocalModelHandler


class DummyModel:
    def fit(self, X, y):
        return self

    def predict(self, X):
        # Predict 250,000 for everything
        return [250000.0] * len(X)


class DummyTrainer(BaseModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(model_name="dummy_model", **kwargs)

    def fit_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        model = DummyModel()
        return model.fit(X_train, y_train)


def _create_synthetic_data():
    return pd.DataFrame(
        {
            "surface": [60.0, 80.0, 100.0, 120.0, 140.0] * 10,
            "rooms": [2, 3, 3, 4, 5] * 10,
            "bathrooms": [1, 1, 2, 2, 3] * 10,
            "elevator": [False, True, True, True, True] * 10,
            "terrace": [False, False, True, True, True] * 10,
            "garage": [False, True, False, True, True] * 10,
            "postal_code": ["36201", "36202", "36203", "36204", "36205"] * 10,
            "price": [150000.0, 200000.0, 250000.0, 300000.0, 350000.0] * 10,
        }
    )


def test_base_trainer_run_lifecycle(tmp_path):
    mock_data_provider = MagicMock()
    mock_data_provider.read_properties_full.return_value = _create_synthetic_data()

    handler = LocalModelHandler("dummy_model", base_path=str(tmp_path))

    trainer = DummyTrainer(
        encoding="onehot",
        test_size=0.2,
        random_state=42,
        data_provider=mock_data_provider,
        model_handler=handler,
    )

    # Mock load_data to return synthetic data directly
    trainer.load_data = lambda: _create_synthetic_data()

    model, features, report, version_id = trainer.run(version="test_v1")

    assert version_id == "test_v1"
    assert "mae" in report
    assert "rmse" in report
    assert "r2_score" in report
    assert report["test_samples"] == 10

    # Test predict sample
    predicted_price = trainer.predict_sample(
        surface=90,
        rooms=3,
        bathrooms=2,
        postal_code="36202",
        elevator=True,
        terrace=False,
        garage=True,
        version="test_v1",
    )
    assert predicted_price == 250000.0
