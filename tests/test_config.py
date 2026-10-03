"""Unit tests for configuration and settings."""

import os
from unittest.mock import patch

from tfg_models.config import Settings


def test_default_settings():
    with patch.dict(os.environ, {}, clear=True):
        settings = Settings()
        assert settings.azure_storage_account == "tfgdatalake"
        assert settings.azure_container == "gold"
        assert settings.properties_table_path == "properties_full"
        assert settings.azure_models_container in ("datalake", "models")
        assert settings.default_data_provider == "azure"
        assert settings.default_model_handler == "azure"


def test_settings_override_via_env():
    with patch.dict(
        os.environ,
        {
            "AZURE_STORAGE_ACCOUNT": "custom_datalake",
            "AZURE_CONTAINER": "silver",
            "DATA_PROVIDER": "local",
            "MODEL_HANDLER": "azure",
        },
    ):
        settings = Settings()
        assert settings.azure_storage_account == "custom_datalake"
        assert settings.azure_container == "silver"
        assert settings.default_data_provider == "local"
        assert settings.default_model_handler == "azure"
