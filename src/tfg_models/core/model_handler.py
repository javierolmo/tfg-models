"""Model handler interfaces and implementations (Azure Blob Storage and Local filesystem) for saving, loading, and versioning models."""

import io
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from tfg_models.config import settings

logger = logging.getLogger(__name__)


def _get_git_commit_hash() -> Optional[str]:
    """Retrieves current git commit hash if available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return None


class ModelHandler(ABC):
    """Abstract base class for saving, versioning, and loading machine learning models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def _prepare_metadata(self, report: dict, version: str) -> dict:
        metadata = dict(report)
        metadata["version"] = version
        metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
        metadata["git_commit"] = _get_git_commit_hash()
        return metadata

    @abstractmethod
    def save_model(
        self,
        model: Any,
        features: Any,
        report: dict,
        version: Optional[str] = None,
    ) -> str:
        """
        Saves model, feature list, and metadata report under a versioned path and as 'latest'.
        Returns the version ID.
        """
        pass

    @abstractmethod
    def load_model(self, version: str = "latest") -> Tuple[Any, Any]:
        """Loads and returns (model, features) for the requested version (defaults to 'latest')."""
        pass

    @abstractmethod
    def get_report(self, version: str = "latest") -> dict:
        """Loads and returns the performance report metadata for the requested version."""
        pass

    @abstractmethod
    def list_versions(self) -> List[str]:
        """Returns a list of available version IDs for this model."""
        pass


class AzureModelHandler(ModelHandler):
    """Production model handler persisting models directly to Azure Blob Storage with versioning."""

    def __init__(
        self,
        model_name: str,
        storage_account: Optional[str] = None,
        connection_string: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: Optional[str] = None,
        client_id: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.storage_account = storage_account or settings.azure_storage_account
        self.connection_string = connection_string or settings.azure_connection_string
        self.account_key = account_key or settings.azure_storage_account_key
        self.container_name = container_name or settings.azure_models_container
        self.client_id = client_id or settings.azure_client_id
        self._blob_service_client: Optional[BlobServiceClient] = None

    def _get_container_client(self):
        if self._blob_service_client is None:
            if self.connection_string:
                self._blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            elif self.account_key:
                account_url = f"https://{self.storage_account}.blob.core.windows.net"
                self._blob_service_client = BlobServiceClient(account_url=account_url, credential=self.account_key)
            else:
                account_url = f"https://{self.storage_account}.blob.core.windows.net"
                credential = (
                    DefaultAzureCredential(managed_identity_client_id=self.client_id)
                    if self.client_id
                    else DefaultAzureCredential()
                )
                self._blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        return self._blob_service_client.get_container_client(self.container_name)

    def _upload_bytes(self, data: bytes, remote_path: str) -> None:
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(remote_path)
        blob_client.upload_blob(data=data, overwrite=True)

    def _download_bytes(self, remote_path: str) -> bytes:
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(remote_path)
        return blob_client.download_blob().readall()

    def save_model(
        self,
        model: Any,
        features: Any,
        report: dict,
        version: Optional[str] = None,
    ) -> str:
        version_id = version or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        metadata = self._prepare_metadata(report, version_id)

        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_bytes = model_buffer.getvalue()

        features_buffer = io.BytesIO()
        joblib.dump(features, features_buffer)
        features_bytes = features_buffer.getvalue()

        report_bytes = json.dumps(metadata, indent=2).encode("utf-8")

        prefixes = [
            f"models/{self.model_name}/{version_id}",
            f"models/{self.model_name}/latest",
        ]

        logger.info("Saving Azure model '%s' (version: %s)", self.model_name, version_id)
        for prefix in prefixes:
            self._upload_bytes(model_bytes, f"{prefix}/model.pkl")
            self._upload_bytes(features_bytes, f"{prefix}/features.pkl")
            self._upload_bytes(report_bytes, f"{prefix}/report.json")

        logger.info("Model '%s' (version %s) successfully uploaded to Azure.", self.model_name, version_id)
        return version_id

    def load_model(self, version: str = "latest") -> Tuple[Any, Any]:
        logger.info("Loading model from Azure for '%s' (version: %s)", self.model_name, version)
        model_data = self._download_bytes(f"models/{self.model_name}/{version}/model.pkl")
        features_data = self._download_bytes(f"models/{self.model_name}/{version}/features.pkl")

        model = joblib.load(io.BytesIO(model_data))
        features = joblib.load(io.BytesIO(features_data))
        return model, features

    def get_report(self, version: str = "latest") -> dict:
        report_data = self._download_bytes(f"models/{self.model_name}/{version}/report.json")
        return json.loads(report_data.decode("utf-8"))

    def list_versions(self) -> List[str]:
        container_client = self._get_container_client()
        prefix = f"models/{self.model_name}/"
        blobs = container_client.list_blobs(name_starts_with=prefix)
        versions = set()
        for blob in blobs:
            parts = blob.name[len(prefix):].split("/")
            if len(parts) >= 2 and parts[0] != "latest":
                versions.add(parts[0])
        return sorted(list(versions), reverse=True)


class LocalModelHandler(ModelHandler):
    """Model handler persisting models to the local filesystem for offline experiments and container runs."""

    def __init__(self, model_name: str, base_path: Optional[str] = None):
        super().__init__(model_name)
        self.base_path = Path(base_path or settings.local_datalake_path)
        self.models_dir = self.base_path / "models" / self.model_name

    def save_model(
        self,
        model: Any,
        features: Any,
        report: dict,
        version: Optional[str] = None,
    ) -> str:
        version_id = version or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        metadata = self._prepare_metadata(report, version_id)

        target_dirs = [
            self.models_dir / version_id,
            self.models_dir / "latest",
        ]

        logger.info("Saving local model '%s' (version: %s)", self.model_name, version_id)
        for directory in target_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, directory / "model.pkl")
            joblib.dump(features, directory / "features.pkl")
            with open(directory / "report.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

        logger.info("Model '%s' (version %s) successfully saved locally.", self.model_name, version_id)
        return version_id

    def load_model(self, version: str = "latest") -> Tuple[Any, Any]:
        target_dir = self.models_dir / version
        model_file = target_dir / "model.pkl"
        features_file = target_dir / "features.pkl"

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found at: {model_file}")

        logger.info("Loading local model from %s", target_dir)
        model = joblib.load(model_file)
        features = joblib.load(features_file)
        return model, features

    def get_report(self, version: str = "latest") -> dict:
        target_dir = self.models_dir / version
        report_file = target_dir / "report.json"
        if not report_file.exists():
            raise FileNotFoundError(f"Report not found at: {report_file}")

        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_versions(self) -> List[str]:
        if not self.models_dir.exists():
            return []
        versions = [
            d.name
            for d in self.models_dir.iterdir()
            if d.is_dir() and d.name not in ("latest", "__pycache__")
        ]
        return sorted(versions, reverse=True)


def get_model_handler(
    model_name: str,
    handler_type: Optional[str] = None,
    **kwargs: Any,
) -> ModelHandler:
    """Factory creating a ModelHandler instance based on settings or explicit argument."""
    handler = (handler_type or settings.default_model_handler).lower()
    if handler == "local":
        return LocalModelHandler(model_name, **kwargs)
    elif handler == "azure":
        return AzureModelHandler(model_name, **kwargs)
    else:
        raise ValueError(
            f"Unknown model handler: '{handler}'. Expected 'azure' or 'local'."
        )


__all__ = [
    "ModelHandler",
    "AzureModelHandler",
    "LocalModelHandler",
    "get_model_handler",
    "_get_git_commit_hash",
]
