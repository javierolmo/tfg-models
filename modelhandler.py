import io
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import joblib
from azure.storage.blob import BlobServiceClient

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelHandler(ABC):
    """Abstract base class for saving and loading machine learning models and metadata."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def save_model(self, model: Any, features: Any, report: dict) -> None:
        """Saves model, feature list, and performance report."""
        pass

    @abstractmethod
    def load_model(self) -> Tuple[Any, Any]:
        """Loads and returns (model, features)."""
        pass


class LocalModelHandler(ModelHandler):
    """Model handler persisting models to the local filesystem."""

    DEFAULT_BASE_PATH = os.path.expanduser("~/local_datalake")

    def __init__(self, model_name: str, base_path: Optional[str] = None):
        super().__init__(model_name)
        self.base_path = base_path or os.environ.get("LOCAL_DATALAKE_PATH", self.DEFAULT_BASE_PATH)

    def save_model(self, model: Any, features: Any, report: dict) -> None:
        model_path = os.path.join(self.base_path, "models", self.model_name)
        os.makedirs(model_path, exist_ok=True)

        logger.info("Saving model to %s", model_path)
        try:
            joblib.dump(model, os.path.join(model_path, "model.pkl"))
            joblib.dump(features, os.path.join(model_path, "features.pkl"))
            with open(os.path.join(model_path, "report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info("Model, features, and report successfully saved locally.")
        except Exception as e:
            logger.error("Error saving the model locally: %s", e)
            raise

    def load_model(self) -> Tuple[Any, Any]:
        model_path = os.path.join(self.base_path, "models", self.model_name)
        logger.info("Loading model from %s", model_path)
        try:
            model = joblib.load(os.path.join(model_path, "model.pkl"))
            features = joblib.load(os.path.join(model_path, "features.pkl"))
            logger.info("Model and features successfully loaded locally.")
            return model, features
        except Exception as e:
            logger.error("Error loading the model locally: %s", e)
            raise


class AzureModelHandler(ModelHandler):
    """Model handler persisting models directly to Azure Blob Storage."""

    DEFAULT_CONTAINER = "datalake"

    def __init__(
        self,
        model_name: str,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.connection_string = (
            connection_string
            or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            or os.environ.get("tfgbs_connection_string")
        )
        self.container_name = (
            container_name
            or os.environ.get("AZURE_STORAGE_CONTAINER")
            or os.environ.get("AZURE_CONTAINER", self.DEFAULT_CONTAINER)
        )
        self._blob_service_client: Optional[BlobServiceClient] = None

    def _get_container_client(self):
        if not self.connection_string:
            raise ValueError(
                "Azure Storage connection string is required. Set AZURE_STORAGE_CONNECTION_STRING "
                "or tfgbs_connection_string environment variable."
            )
        if self._blob_service_client is None:
            self._blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        return self._blob_service_client.get_container_client(self.container_name)

    def _upload_bytes(self, data: bytes, remote_path: str) -> None:
        logger.info("Uploading %d bytes to Azure Blob Storage at %s", len(data), remote_path)
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(remote_path)
            blob_client.upload_blob(data=data, overwrite=True)
            logger.info("Upload completed: %s", remote_path)
        except Exception as e:
            logger.error("Error uploading to Azure Blob Storage: %s", e)
            raise

    def _download_bytes(self, remote_path: str) -> bytes:
        logger.info("Downloading from Azure Blob Storage: %s", remote_path)
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(remote_path)
            data = blob_client.download_blob().readall()
            logger.info("Downloaded %d bytes from %s", len(data), remote_path)
            return data
        except Exception as e:
            logger.error("Error downloading from Azure Blob Storage: %s", e)
            raise

    def save_model(self, model: Any, features: Any, report: dict) -> None:
        try:
            # Serialize model to in-memory bytes
            model_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            self._upload_bytes(model_buffer.getvalue(), f"models/{self.model_name}/model.pkl")

            # Serialize features to in-memory bytes
            features_buffer = io.BytesIO()
            joblib.dump(features, features_buffer)
            self._upload_bytes(features_buffer.getvalue(), f"models/{self.model_name}/features.pkl")

            # Serialize report to in-memory bytes
            report_bytes = json.dumps(report, indent=2).encode("utf-8")
            self._upload_bytes(report_bytes, f"models/{self.model_name}/report.json")

            logger.info("Model, features, and report successfully uploaded to Azure.")
        except Exception as e:
            logger.error("Error saving the model to Azure: %s", e)
            raise

    def load_model(self) -> Tuple[Any, Any]:
        logger.info("Loading model from Azure for %s", self.model_name)
        try:
            model_data = self._download_bytes(f"models/{self.model_name}/model.pkl")
            model = joblib.load(io.BytesIO(model_data))

            features_data = self._download_bytes(f"models/{self.model_name}/features.pkl")
            features = joblib.load(io.BytesIO(features_data))

            logger.info("Model and features successfully loaded from Azure.")
            return model, features
        except Exception as e:
            logger.error("Error loading the model from Azure: %s", e)
            raise
