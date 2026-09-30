"""Unit tests for model handlers (LocalModelHandler in test helpers and AzureModelHandler in core)."""

import io
import json
import joblib
import pytest
from unittest.mock import MagicMock, patch

from tfg_models.core.model_handler import AzureModelHandler
from tests.helpers.local_model_handler import LocalModelHandler


class DummyModel:
    def __init__(self, val=42):
        self.val = val

    def predict(self, X):
        return [self.val] * len(X)


class TestLocalModelHandler:
    def test_save_and_load_model(self, tmp_path):
        handler = LocalModelHandler("test_model", base_path=str(tmp_path))
        model = DummyModel(99)
        features = ["surface", "rooms"]
        report = {"mae": 100.0, "rmse": 200.0, "r2_score": 0.5}

        version_id = handler.save_model(model, features, report, version="v1.0")
        assert version_id == "v1.0"

        # Check files exist
        assert (tmp_path / "models" / "test_model" / "v1.0" / "model.pkl").exists()
        assert (tmp_path / "models" / "test_model" / "latest" / "model.pkl").exists()

        # Load specific version
        loaded_model, loaded_features = handler.load_model(version="v1.0")
        assert loaded_model.val == 99
        assert loaded_features == features

        # Load latest
        loaded_latest, _ = handler.load_model(version="latest")
        assert loaded_latest.val == 99

    def test_get_report(self, tmp_path):
        handler = LocalModelHandler("test_model", base_path=str(tmp_path))
        model = DummyModel()
        report = {"mae": 50.0, "r2_score": 0.8}
        handler.save_model(model, ["feat1"], report, version="v2.0")

        loaded_report = handler.get_report(version="v2.0")
        assert loaded_report["mae"] == 50.0
        assert loaded_report["r2_score"] == 0.8
        assert loaded_report["version"] == "v2.0"
        assert "saved_at" in loaded_report

    def test_list_versions(self, tmp_path):
        handler = LocalModelHandler("test_model", base_path=str(tmp_path))
        model = DummyModel()
        handler.save_model(model, [], {}, version="20260101_000000")
        handler.save_model(model, [], {}, version="20260102_000000")

        versions = handler.list_versions()
        assert "20260102_000000" in versions
        assert "20260101_000000" in versions
        assert "latest" not in versions

    def test_load_nonexistent_model_raises_error(self, tmp_path):
        handler = LocalModelHandler("nonexistent", base_path=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            handler.load_model("v999")


class TestAzureModelHandler:
    def test_init_without_connection_string_raises_on_access(self):
        handler = AzureModelHandler("test_azure")
        handler.connection_string = None
        with pytest.raises(ValueError, match="Azure Storage connection string is required"):
            handler._get_container_client()

    @patch("tfg_models.core.model_handler.BlobServiceClient")
    def test_save_model(self, mock_blob_service_cls):
        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_blob = MagicMock()

        mock_blob_service_cls.from_connection_string.return_value = mock_service
        mock_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob

        handler = AzureModelHandler(
            "test_model",
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake;EndpointSuffix=core.windows.net",
            container_name="models",
        )

        model = DummyModel(123)
        features = ["f1", "f2"]
        report = {"mae": 10.0}

        version_id = handler.save_model(model, features, report, version="v1.0")
        assert version_id == "v1.0"
        assert mock_blob.upload_blob.called

    @patch("tfg_models.core.model_handler.BlobServiceClient")
    def test_load_model(self, mock_blob_service_cls):
        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_blob = MagicMock()

        mock_blob_service_cls.from_connection_string.return_value = mock_service
        mock_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob

        # Create serialized mock model and features
        buf_model = io.BytesIO()
        joblib.dump(DummyModel(777), buf_model)
        buf_features = io.BytesIO()
        joblib.dump(["feat_a", "feat_b"], buf_features)

        mock_download_model = MagicMock()
        mock_download_model.readall.return_value = buf_model.getvalue()
        mock_download_features = MagicMock()
        mock_download_features.readall.return_value = buf_features.getvalue()

        mock_blob.download_blob.side_effect = [mock_download_model, mock_download_features]

        handler = AzureModelHandler("test_model", connection_string="fake_conn")
        loaded_model, loaded_features = handler.load_model(version="v1")
        assert loaded_model.val == 777
        assert loaded_features == ["feat_a", "feat_b"]

    @patch("tfg_models.core.model_handler.BlobServiceClient")
    def test_get_report(self, mock_blob_service_cls):
        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_blob = MagicMock()

        mock_blob_service_cls.from_connection_string.return_value = mock_service
        mock_service.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob

        report_json = json.dumps({"mae": 45000.0, "version": "v1"}).encode("utf-8")
        mock_download = MagicMock()
        mock_download.readall.return_value = report_json
        mock_blob.download_blob.return_value = mock_download

        handler = AzureModelHandler("test_model", connection_string="fake_conn")
        report = handler.get_report(version="v1")
        assert report["mae"] == 45000.0
        assert report["version"] == "v1"

    @patch("tfg_models.core.model_handler.BlobServiceClient")
    def test_list_versions(self, mock_blob_service_cls):
        mock_service = MagicMock()
        mock_container = MagicMock()

        blob1 = MagicMock()
        blob1.name = "models/test_model/20260901_120000/model.pkl"
        blob2 = MagicMock()
        blob2.name = "models/test_model/20260902_120000/report.json"
        blob_latest = MagicMock()
        blob_latest.name = "models/test_model/latest/model.pkl"

        mock_container.list_blobs.return_value = [blob1, blob2, blob_latest]
        mock_blob_service_cls.from_connection_string.return_value = mock_service
        mock_service.get_container_client.return_value = mock_container

        handler = AzureModelHandler(
            "test_model",
            connection_string="fake_connection_string",
        )

        versions = handler.list_versions()
        assert "20260902_120000" in versions
        assert "20260901_120000" in versions
        assert "latest" not in versions
