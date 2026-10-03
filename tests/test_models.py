"""Unit tests for concrete model trainers (LinearRegression and LightGBM)."""

import pandas as pd
import pytest

from tfg_models.core.model_handler import LocalModelHandler
from tfg_models.models import (
    MODEL_REGISTRY,
    LightGBMTrainer,
    LinearRegressionTrainer,
    get_model_trainer,
)


def _create_train_dataset():
    # 100 rows with clear correlation
    data = []
    for i in range(100):
        surface = 50.0 + (i % 50) * 2.0
        rooms = int(surface // 25)
        bathrooms = int(surface // 50)
        pc = "36201" if i % 2 == 0 else "36211"
        price = surface * 2500 + rooms * 10000 + (20000 if pc == "36211" else 0)
        data.append(
            {
                "surface": surface,
                "rooms": rooms,
                "bathrooms": bathrooms,
                "elevator": i % 2 == 0,
                "terrace": i % 3 == 0,
                "garage": i % 4 == 0,
                "postal_code": pc,
                "price": price,
            }
        )
    return pd.DataFrame(data)


def test_get_model_trainer():
    lr = get_model_trainer("linear_regression")
    assert isinstance(lr, LinearRegressionTrainer)

    lgb = get_model_trainer("lightgbm")
    assert isinstance(lgb, LightGBMTrainer)

    with pytest.raises(ValueError, match="Unknown model"):
        get_model_trainer("unsupported_model")


def test_linear_regression_trainer_run_and_predict(tmp_path):
    handler = LocalModelHandler("linear_regression", base_path=str(tmp_path))
    trainer = LinearRegressionTrainer(model_handler=handler)
    trainer.load_data = _create_train_dataset

    model, features, report, version = trainer.run(version="lr_v1")

    assert version == "lr_v1"
    assert report["mae"] < 50000.0  # Should fit well on synthetic linear data
    assert report["r2_score"] > 0.8

    pred = trainer.predict_sample(
        surface=75.0,
        rooms=3,
        bathrooms=1,
        postal_code="36201",
        version="lr_v1",
    )
    assert pred > 100000.0


def test_lightgbm_trainer_run_and_predict(tmp_path):
    handler = LocalModelHandler("lightgbm", base_path=str(tmp_path))
    trainer = LightGBMTrainer(n_estimators=50, model_handler=handler)
    trainer.load_data = _create_train_dataset

    model, features, report, version = trainer.run(version="lgb_v1")

    assert version == "lgb_v1"
    assert "feature_importances" in report
    assert "surface" in report["feature_importances"]
    assert report["feature_importances"]["surface"] > 0

    pred = trainer.predict_sample(
        surface=85.0,
        rooms=3,
        bathrooms=2,
        postal_code="36211",
        version="lgb_v1",
    )
    assert pred > 100000.0
