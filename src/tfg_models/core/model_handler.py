"""Model handler interfaces and implementations for saving, loading, and versioning models."""

import io
import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib
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


class LocalModelHandler(ModelHandler):
    """Model handler persisting models to the local filesystem with versioning."""

    def __init__(self, model_name: str, base_path: Optional[str] = None):
        super().__init__(model_name)
        self.base_path = Path(base_path or settings.local_datalake_path)
        self.models_dir = self.base_path / "models" / self.model_name

    def _prepare_metadata(self, report: dict, version: str) -> dict:
        metadata = dict(report)
        metadata["version"] = version
        metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
        metadata["git_commit"] = _get_git_commit_hash()
        return metadata

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
            self.models_dir,  # Direct folder for backward compatibility
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
        if not (target_dir / "model.pkl").exists() and version == "latest":
            target_dir = self.models_dir  # Fallback to direct dir

        if not (target_dir / "model.pkl").exists():
            raise FileNotFoundError(f"Model not found at: {target_dir / 'model.pkl'}")

        logger.info("Loading local model from %s", target_dir)
        model = joblib.load(target_dir / "model.pkl")
        features = joblib.load(target_dir / "features.pkl")
        return model, features

    def get_report(self, version: str = "latest") -> dict:
        target_dir = self.models_dir / version
        if not (target_dir / "report.json").exists() and version == "latest":
            target_dir = self.models_dir

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


class AzureModelHandler(ModelHandler):
    """Model handler persisting models directly to Azure Blob Storage with versioning."""

    def __init__(
        self,
        model_name: str,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.connection_string = connection_string or settings.azure_connection_string
        self.container_name = container_name or settings.azure_models_container
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
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(remote_path)
        blob_client.upload_blob(data=data, overwrite=True)

    def _download_bytes(self, remote_path: str) -> bytes:
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(remote_path)
        return blob_client.download_blob().readall()

    def _prepare_metadata(self, report: dict, version: str) -> dict:
        metadata = dict(report)
        metadata["version"] = version
        metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
        metadata["git_commit"] = _get_git_commit_hash()
        return metadata

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
            f"models/{self.model_name}",  # Backward-compatible root
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
        try:
            model_data = self._download_bytes(f"models/{self.model_name}/{version}/model.pkl")
            features_data = self._download_bytes(f"models/{self.model_name}/{version}/features.pkl")
        except Exception:
            if version == "latest":
                # Fallback to direct path
                model_data = self._download_bytes(f"models/{self.model_name}/model.pkl")
                features_data = self._download_bytes(f"models/{self.model_name}/features.pkl")
            else:
                raise

        model = joblib.load(io.BytesIO(model_data))
        features = joblib.load(io.BytesIO(features_data))
        return model, features

    def get_report(self, version: str = "latest") -> dict:
        try:
            report_data = self._download_bytes(f"models/{self.model_name}/{version}/report.json")
        except Exception:
            if version == "latest":
                report_data = self._download_bytes(f"models/{self.model_name}/report.json")
            else:
                raise
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
