"""Unit tests for concrete model trainers (LinearRegression, LightGBM, and NeuralNetwork)."""

import pandas as pd
import pytest

from tfg_models.core.model_handler import LocalModelHandler
from tfg_models.models import (
    MODEL_REGISTRY,
    LightGBMTrainer,
    LinearRegressionTrainer,
    NeuralNetworkTrainer,
    get_model_trainer,
)
from tfg_models.models.neural_network import PyTorchTabularRegressor


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

    nn = get_model_trainer("neural_network")
    assert isinstance(nn, NeuralNetworkTrainer)

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


def test_neural_network_trainer_run_and_predict(tmp_path):
    handler = LocalModelHandler("neural_network", base_path=str(tmp_path))
    trainer = NeuralNetworkTrainer(
        epochs=5,
        batch_size=32,
        embedding_dim=8,
        hidden_dims=[64, 32],
        model_handler=handler,
    )
    trainer.load_data = _create_train_dataset

    model, features, report, version = trainer.run(version="nn_v1")

    assert version == "nn_v1"
    assert "architecture" in report
    assert report["architecture"] == "Tabular MLP with Entity Embeddings"
    assert report["embedding_dim"] == 8
    assert report["epochs"] == 5
    assert report["postal_codes_learned"] == 2
    assert "mae" in report
    assert report["mae"] > 0

    # Predict with a known postal code
    pred_known = trainer.predict_sample(
        surface=80.0,
        rooms=3,
        bathrooms=1,
        postal_code="36201",
        version="nn_v1",
    )
    assert pred_known > 50000.0

    # Predict with an unknown postal code to verify <unk> handling
    pred_unknown = trainer.predict_sample(
        surface=80.0,
        rooms=3,
        bathrooms=1,
        postal_code="99999",
        version="nn_v1",
    )
    assert pred_unknown > 50000.0


def test_neural_network_unfitted_predict():
    regressor = PyTorchTabularRegressor()
    with pytest.raises(RuntimeError, match="Model must be fitted"):
        regressor.predict(pd.DataFrame([{"surface": 100.0}]))
