"""Unit tests for data providers."""

import pytest
from unittest.mock import MagicMock, patch

from tfg_models.data.providers import AzureDataProvider, LocalDataProvider


def test_local_data_provider_read(tmp_path):
    mock_spark = MagicMock()
    mock_df = MagicMock()
    mock_spark.read.format.return_value.load.return_value = mock_df

    provider = LocalDataProvider(
        base_path=str(tmp_path),
        table_path="properties_full",
        spark_session=mock_spark,
    )

    df = provider.read_properties_full()
    assert df == mock_df
    assert mock_spark.read.format.called


def test_azure_data_provider_read():
    mock_spark = MagicMock()
    mock_df = MagicMock()
    mock_spark.read.format.return_value.load.return_value = mock_df

    provider = AzureDataProvider(
        storage_account="testacc",
        container="gold",
        table_path="properties_full",
        account_key="fake_key",
        spark_session=mock_spark,
    )

    df = provider.read_properties_full()
    assert df == mock_df
    assert "abfss://gold@testacc.dfs.core.windows.net/properties_full" in provider.base_url


def test_read_properties_snapshot_with_date():
    mock_spark = MagicMock()
    mock_df = MagicMock()
    mock_df.columns = ["property_id", "load_date"]
    mock_filtered_df = MagicMock()
    mock_df.filter.return_value = mock_filtered_df

    provider = LocalDataProvider(base_path="/tmp", spark_session=mock_spark)
    provider.read_properties_full = MagicMock(return_value=mock_df)

    from datetime import datetime

    res = provider.read_properties_snapshot(date=datetime(2026, 9, 30))
    assert res == mock_filtered_df
    mock_df.filter.assert_called_with("load_date = '2026-09-30'")
