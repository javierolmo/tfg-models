"""Local filesystem model handler for unit testing and offline experimentation."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib

from tfg_models.core.model_handler import ModelHandler, _get_git_commit_hash

logger = logging.getLogger(__name__)


class LocalModelHandler(ModelHandler):
    """Model handler persisting models to the local filesystem for tests and fixtures."""

    def __init__(self, model_name: str, base_path: Optional[str] = None):
        super().__init__(model_name)
        self.base_path = Path(base_path or (Path.home() / "local_datalake"))
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
            self.models_dir,
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
            target_dir = self.models_dir

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
