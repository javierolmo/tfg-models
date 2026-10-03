"""Unit tests for the Inference and Training HTTP APIs."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from tfg_models.api.inference import app as inference_app, _MODEL_CACHE
from tfg_models.api.training import app as training_app


class DummyModel:
    def predict(self, X):
        return [250000.0] * len(X)


@pytest.fixture(autouse=True)
def disable_model_preload_in_tests(monkeypatch):
    """Prevents automatic network calls to Azure IMDS during test lifespan execution."""
    monkeypatch.setenv("PRELOAD_MODEL", "none")


@pytest.fixture
def inference_client():
    _MODEL_CACHE.clear()
    with TestClient(inference_app) as client:
        yield client
    _MODEL_CACHE.clear()


@pytest.fixture
def training_client():
    with TestClient(training_app) as client:
        yield client


class TestInferenceApi:
    def test_health(self, inference_client):
        response = inference_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "tfg-models-inference"

    def test_list_models(self, inference_client):
        response = inference_client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "lightgbm" in data["registered_models"]
        assert "linear_regression" in data["registered_models"]
        assert "neural_network" in data["registered_models"]

    def test_list_versions_unknown_model(self, inference_client):
        response = inference_client.get("/models/unknown_model/versions")
        assert response.status_code == 404

    @patch("tfg_models.api.inference.get_model_handler")
    def test_list_versions_success(self, mock_get_handler, inference_client):
        mock_handler = MagicMock()
        mock_handler.list_versions.return_value = ["20261001_100000", "20261002_100000"]
        mock_get_handler.return_value = mock_handler

        response = inference_client.get("/models/lightgbm/versions")
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "lightgbm"
        assert len(data["versions"]) == 2

    @patch("tfg_models.api.inference.get_model_handler")
    def test_predict_success_lightgbm(self, mock_get_handler, inference_client):
        mock_handler = MagicMock()
        mock_handler.load_model.return_value = (DummyModel(), ["surface", "rooms", "postal_code"])
        mock_get_handler.return_value = mock_handler

        payload = {
            "model": "lightgbm",
            "version": "latest",
            "surface": 85.5,
            "rooms": 3,
            "bathrooms": 2,
            "postal_code": "36211",
            "elevator": True,
            "terrace": False,
            "garage": True,
        }
        response = inference_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "lightgbm"
        assert data["estimated_price"] == 250000.0
        assert data["currency"] == "EUR"
        assert data["inputs"]["surface"] == 85.5

    @patch("tfg_models.api.inference.get_model_handler")
    def test_predict_success_linear_regression(self, mock_get_handler, inference_client):
        mock_handler = MagicMock()
        mock_handler.load_model.return_value = (
            DummyModel(),
            ["surface", "rooms", "postal_code_36211"],
        )
        mock_get_handler.return_value = mock_handler

        payload = {
            "model": "linear_regression",
            "version": "latest",
            "surface": 110.0,
            "rooms": 4,
            "bathrooms": 2,
            "postal_code": "36211",
            "elevator": True,
            "terrace": True,
            "garage": False,
        }
        response = inference_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "linear_regression"
        assert data["estimated_price"] == 250000.0

    @patch("tfg_models.api.inference.get_model_handler")
    def test_predict_success_neural_network(self, mock_get_handler, inference_client):
        mock_handler = MagicMock()
        mock_handler.load_model.return_value = (
            DummyModel(),
            ["surface", "rooms", "bathrooms", "elevator", "terrace", "garage", "postal_code"],
        )
        mock_get_handler.return_value = mock_handler

        payload = {
            "model": "neural_network",
            "version": "latest",
            "surface": 95.0,
            "rooms": 3,
            "bathrooms": 2,
            "postal_code": "36211",
            "elevator": True,
            "terrace": False,
            "garage": True,
        }
        response = inference_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "neural_network"
        assert data["estimated_price"] == 250000.0

    def test_predict_validation_error(self, inference_client):
        # Surface must be > 0
        payload = {
            "model": "lightgbm",
            "surface": -10.0,
            "rooms": 2,
            "bathrooms": 1,
            "postal_code": "36201",
        }
        response = inference_client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_unknown_model(self, inference_client):
        payload = {
            "model": "non_existent_model",
            "surface": 90.0,
            "rooms": 3,
            "bathrooms": 2,
            "postal_code": "36201",
        }
        response = inference_client.post("/predict", json=payload)
        assert response.status_code == 404

    @patch("tfg_models.api.inference.load_and_cache_model")
    def test_lifespan_warmup_execution(self, mock_load, monkeypatch):
        monkeypatch.setenv("PRELOAD_MODEL", "lightgbm")
        monkeypatch.setenv("PRELOAD_VERSION", "latest")
        with TestClient(inference_app):
            mock_load.assert_called_once_with("lightgbm", "latest")


class TestTrainingApi:
    def test_health(self, training_client):
        response = training_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "tfg-models-training"

    @patch("tfg_models.api.training.get_model_trainer")
    def test_train_sync_success(self, mock_get_trainer, training_client):
        mock_trainer = MagicMock()
        mock_trainer.run.return_value = (
            DummyModel(),
            ["surface"],
            {"mae": 15000.0, "rmse": 20000.0, "r2_score": 0.85},
            "20261003_120000",
        )
        mock_get_trainer.return_value = mock_trainer

        payload = {"model": "lightgbm", "background": False}
        response = training_client.post("/train", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert len(data["results"]) == 1
        assert data["results"][0]["model"] == "lightgbm"
        assert data["results"][0]["version"] == "20261003_120000"
        assert data["results"][0]["metrics"]["mae"] == 15000.0

    def test_train_invalid_model(self, training_client):
        payload = {"model": "unsupported_model"}
        response = training_client.post("/train", json=payload)
        assert response.status_code == 400

    @patch("tfg_models.api.training._execute_training")
    def test_train_background(self, mock_exec, training_client):
        payload = {"model": "lightgbm", "background": True}
        response = training_client.post("/train", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "job_id" in data
        job_id = data["job_id"]

        # Check job status endpoint
        job_resp = training_client.get(f"/train/jobs/{job_id}")
        assert job_resp.status_code == 200
        job_data = job_resp.json()
        assert job_data["status"] in ["pending", "running", "completed"]

    def test_job_not_found(self, training_client):
        response = training_client.get("/train/jobs/unknown-id")
        assert response.status_code == 404

    @patch("tfg_models.api.training.get_model_handler")
    def test_compare_models(self, mock_get_handler, training_client):
        mock_handler = MagicMock()
        mock_handler.get_report.return_value = {
            "version": "latest",
            "mae": 12000.0,
            "rmse": 18000.0,
            "r2_score": 0.88,
        }
        mock_get_handler.return_value = mock_handler

        response = training_client.get("/compare")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) >= 2

    @patch("tfg_models.api.training.get_model_handler")
    def test_get_model_report(self, mock_get_handler, training_client):
        mock_handler = MagicMock()
        mock_handler.get_report.return_value = {
            "model_name": "lightgbm",
            "mae": 14000.0,
            "r2_score": 0.84,
        }
        mock_get_handler.return_value = mock_handler

        response = training_client.get("/models/lightgbm/report")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "lightgbm"
        assert data["mae"] == 14000.0
