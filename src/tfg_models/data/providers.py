"""Data provider interfaces and implementations for loading property data."""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import pandas as pd

try:
    import pyarrow.dataset as ds
    from pyarrow.fs import AzureFileSystem
except ImportError:  # pragma: no cover
    ds = None
    AzureFileSystem = None

from tfg_models.config import settings

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for property data providers."""

    @abstractmethod
    def read_properties_full(self) -> pd.DataFrame:
        """Reads the full consolidated properties dataset (properties_full)."""
        pass

    def read_properties_snapshot(self, date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Reads property data. If date is provided and load_date is available,
        filters by date. Otherwise returns the full dataset.
        Maintained for backwards-compatibility.
        """
        df = self.read_properties_full()
        if date and "load_date" in df.columns:
            date_str = date.strftime("%Y-%m-%d")
            logger.info("Filtering properties_full by load_date = '%s'", date_str)
            return df[df["load_date"] == date_str]
        return df


class AzureDataProvider(DataProvider):
    """Data provider reading property data directly from Azure Data Lake Storage Gen2 (Parquet)."""

    def __init__(
        self,
        storage_account: Optional[str] = None,
        container: Optional[str] = None,
        table_path: Optional[str] = None,
        account_key: Optional[str] = None,
        filesystem: Optional[Any] = None,
    ):
        if ds is None or AzureFileSystem is None:
            raise ImportError(
                "pyarrow is required for AzureDataProvider. "
                "Install training dependencies with: pip install 'tfg-models[train]'"
            )

        self.storage_account = storage_account or settings.azure_storage_account
        self.container = container or settings.azure_container
        self.table_path = table_path or settings.properties_table_path
        self.account_key = account_key or settings.azure_storage_account_key

        # If account_key is missing but connection string is provided, attempt extraction
        if not self.account_key and settings.azure_storage_connection_string:
            parts = dict(
                item.split("=", 1)
                for item in settings.azure_storage_connection_string.split(";")
                if "=" in item
            )
            self.account_key = parts.get("AccountKey")
            if not self.storage_account:
                self.storage_account = parts.get("AccountName")

        if filesystem:
            self.fs = filesystem
        elif self.account_key:
            self.fs = AzureFileSystem(
                account_name=self.storage_account,
                account_key=self.account_key,
            )
        else:
            self.fs = AzureFileSystem(account_name=self.storage_account)

        self.full_path = f"{self.container}/{self.table_path}".strip("/")
        logger.info(
            "Initializing AzureDataProvider for Data Lake path: %s (account: %s)",
            self.full_path,
            self.storage_account,
        )

    def read_properties_full(self) -> pd.DataFrame:
        """Reads properties_full directly from Azure Data Lake Storage Gen2 using native PyArrow."""
        logger.info("Reading properties_full from Azure Data Lake: %s", self.full_path)
        try:
            dataset = ds.dataset(self.full_path, filesystem=self.fs, format="parquet")
            table = dataset.to_table()
            df = table.to_pandas()
            logger.info("Properties successfully loaded from Azure Data Lake (%d rows).", len(df))
            return df
        except Exception as e:
            logger.error("Failed to read properties_full from Azure Data Lake (%s): %s", self.full_path, e)
            raise


class LocalDataProvider(DataProvider):
    """Data provider reading property data from local parquet storage."""

    def __init__(
        self,
        base_path: Optional[str] = None,
        table_path: Optional[str] = None,
    ):
        self.base_path = str(base_path or settings.local_datalake_path)
        self.table_path = table_path or settings.properties_table_path
        logger.info("Initializing LocalDataProvider at base path: %s", self.base_path)

    def read_properties_full(self) -> pd.DataFrame:
        """Reads properties_full from local filesystem using native PyArrow."""
        if ds is None:
            raise ImportError(
                "pyarrow is required for LocalDataProvider. "
                "Install training dependencies with: pip install 'tfg-models[train]'"
            )

        gold_path = os.path.join(self.base_path, "gold", self.table_path)
        direct_path = os.path.join(self.base_path, self.table_path)

        file_path = gold_path if os.path.exists(gold_path) else direct_path
        logger.info("Reading properties_full from local storage: %s", file_path)
        try:
            dataset = ds.dataset(file_path, format="parquet")
            table = dataset.to_table()
            df = table.to_pandas()
            logger.info("Properties successfully loaded from local storage (%d rows).", len(df))
            return df
        except Exception as e:
            logger.error("Failed to read properties_full from local storage (%s): %s", file_path, e)
            raise
