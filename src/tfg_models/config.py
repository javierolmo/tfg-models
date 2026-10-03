"""Centralized configuration module for tfg-models."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Application settings and environment variable bindings."""

    # Data Lake / Azure Storage Configuration
    azure_storage_account: str = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_ACCOUNT", "tfgdatalake")
    )
    azure_container: str = field(
        default_factory=lambda: os.environ.get("AZURE_CONTAINER", "gold")
    )
    properties_table_path: str = field(
        default_factory=lambda: os.environ.get("PROPERTIES_PATH", "properties_full")
    )
    azure_models_container: str = field(
        default_factory=lambda: (
            os.environ.get("AZURE_STORAGE_CONTAINER")
            or os.environ.get("AZURE_MODELS_CONTAINER", "datalake")
        )
    )
    azure_storage_account_key: Optional[str] = field(
        default_factory=lambda: (
            os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            or os.environ.get("AZURE_STORAGE_KEY")
            or os.environ.get("TFGBS_KEY")
        )
    )
    azure_connection_string: Optional[str] = field(
        default_factory=lambda: (
            os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            or os.environ.get("tfgbs_connection_string")
        )
    )

    # Local Storage Configuration
    local_datalake_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("LOCAL_DATALAKE_PATH", str(Path.home() / "local_datalake"))
        ).resolve()
    )

    # Defaults
    default_data_provider: str = field(
        default_factory=lambda: os.environ.get("DATA_PROVIDER", "azure").lower()
    )
    default_model_handler: str = field(
        default_factory=lambda: os.environ.get("MODEL_HANDLER", "azure").lower()
    )
    default_test_size: float = 0.2
    default_random_state: int = 42


# Global settings singleton instance
settings = Settings()
