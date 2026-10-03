"""Unit tests for data providers."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tfg_models.data.providers import AzureDataProvider, LocalDataProvider


def test_local_data_provider_read(tmp_path):
    # Create real parquet file on local disk
    sample_df = pd.DataFrame(
        {
            "property_id": [1, 2],
            "surface": [80.0, 95.0],
            "price": [200000.0, 310000.0],
        }
    )
    table_dir = tmp_path / "properties_full"
    os.makedirs(table_dir, exist_ok=True)
    sample_df.to_parquet(table_dir / "data.parquet", index=False)

    provider = LocalDataProvider(
        base_path=str(tmp_path),
        table_path="properties_full",
    )

    df = provider.read_properties_full()
    assert len(df) == 2
    assert list(df["property_id"]) == [1, 2]
    assert list(df["surface"]) == [80.0, 95.0]


def test_azure_data_provider_read():
    mock_fs = MagicMock()
    mock_dataset = MagicMock()
    sample_df = pd.DataFrame({"property_id": [1], "price": [150000.0]})
    mock_table = MagicMock()
    mock_table.to_pandas.return_value = sample_df
    mock_dataset.to_table.return_value = mock_table

    with patch("pyarrow.dataset.dataset", return_value=mock_dataset) as mock_ds:
        provider = AzureDataProvider(
            storage_account="testacc",
            container="gold",
            table_path="properties_full",
            account_key="fake_key",
            filesystem=mock_fs,
        )

        df = provider.read_properties_full()
        assert len(df) == 1
        assert df["price"].iloc[0] == 150000.0
        mock_ds.assert_called_once_with("gold/properties_full", filesystem=mock_fs, format="parquet")


def test_azure_data_provider_connection_string_extraction():
    with patch("tfg_models.data.providers.settings") as mock_settings:
        mock_settings.azure_storage_account = None
        mock_settings.azure_container = "gold"
        mock_settings.properties_table_path = "properties_full"
        mock_settings.azure_storage_account_key = None
        mock_settings.azure_connection_string = (
            "DefaultEndpointsProtocol=https;AccountName=extracted_acc;AccountKey=extracted_key;EndpointSuffix=core.windows.net"
        )
        with patch("tfg_models.data.providers.AzureFileSystem") as mock_afs:
            provider = AzureDataProvider()
            assert provider.storage_account == "extracted_acc"
            assert provider.account_key == "extracted_key"
            mock_afs.assert_called_once_with(account_name="extracted_acc", account_key="extracted_key")


def test_azure_data_provider_managed_identity():
    with patch("tfg_models.data.providers.settings") as mock_settings:
        mock_settings.azure_storage_account = "managed_acc"
        mock_settings.azure_container = "gold"
        mock_settings.properties_table_path = "properties_full"
        mock_settings.azure_storage_account_key = None
        mock_settings.azure_connection_string = None
        mock_settings.azure_client_id = "test-client-id"
        with patch("tfg_models.data.providers.AzureFileSystem") as mock_afs:
            provider = AzureDataProvider()
            assert provider.storage_account == "managed_acc"
            assert provider.client_id == "test-client-id"
            mock_afs.assert_called_once_with(account_name="managed_acc", client_id="test-client-id")
