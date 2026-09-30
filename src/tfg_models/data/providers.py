"""Data provider interfaces and implementations for loading property data."""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

from tfg_models.config import settings

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for property data providers."""

    def __init__(self, spark_session: Optional[SparkSession] = None):
        self.spark = spark_session

    @abstractmethod
    def read_properties_full(self) -> DataFrame:
        """Reads the full consolidated properties dataset (properties_full)."""
        pass

    def read_properties_snapshot(self, date: Optional[datetime] = None) -> DataFrame:
        """
        Reads property data. If date is provided and load_date is available,
        filters by date. Otherwise returns the full dataset.
        Maintained for backwards-compatibility.
        """
        df = self.read_properties_full()
        if date and "load_date" in df.columns:
            date_str = date.strftime("%Y-%m-%d")
            logger.info("Filtering properties_full by load_date = '%s'", date_str)
            return df.filter(f"load_date = '{date_str}'")
        return df


class AzureDataProvider(DataProvider):
    """Data provider reading property data directly from Azure Data Lake Storage Gen2 (Parquet)."""

    def __init__(
        self,
        storage_account: Optional[str] = None,
        container: Optional[str] = None,
        table_path: Optional[str] = None,
        account_key: Optional[str] = None,
        spark_session: Optional[SparkSession] = None,
    ):
        self.storage_account = storage_account or settings.azure_storage_account
        self.container = container or settings.azure_container
        self.table_path = table_path or settings.properties_table_path
        self.account_key = account_key or settings.azure_storage_account_key

        if not spark_session:
            builder = SparkSession.builder.appName("tfg-models")
            builder = builder.config("spark.jars.packages", settings.spark_jars_packages)
            if self.account_key:
                builder = builder.config(
                    f"spark.hadoop.fs.azure.account.auth.type.{self.storage_account}.dfs.core.windows.net",
                    "SharedKey",
                ).config(
                    f"spark.hadoop.fs.azure.account.key.{self.storage_account}.dfs.core.windows.net",
                    self.account_key,
                )
            spark_session = builder.getOrCreate()
        else:
            if self.account_key:
                hadoop_conf = spark_session.sparkContext._jsc.hadoopConfiguration()
                hadoop_conf.set(
                    f"fs.azure.account.auth.type.{self.storage_account}.dfs.core.windows.net",
                    "SharedKey",
                )
                hadoop_conf.set(
                    f"fs.azure.account.key.{self.storage_account}.dfs.core.windows.net",
                    self.account_key,
                )

        super().__init__(spark_session=spark_session)
        self.base_url = f"abfss://{self.container}@{self.storage_account}.dfs.core.windows.net/{self.table_path}"
        logger.info("Initializing AzureDataProvider with Data Lake at: %s", self.base_url)

    def read_properties_full(self) -> DataFrame:
        """Reads properties_full directly from Azure Data Lake Storage Gen2."""
        logger.info("Reading properties_full from Azure Data Lake: %s", self.base_url)
        try:
            df = self.spark.read.format("parquet").load(self.base_url)
            logger.info("Properties successfully loaded from Azure Data Lake.")
            return df
        except Exception as e:
            logger.error("Failed to read properties_full from Azure Data Lake (%s): %s", self.base_url, e)
            raise


class LocalDataProvider(DataProvider):
    """Data provider reading property data from local parquet storage."""

    def __init__(
        self,
        base_path: Optional[str] = None,
        table_path: Optional[str] = None,
        spark_session: Optional[SparkSession] = None,
    ):
        if not spark_session:
            spark_session = SparkSession.builder.appName("tfg-models").getOrCreate()

        super().__init__(spark_session=spark_session)
        self.base_path = str(base_path or settings.local_datalake_path)
        self.table_path = table_path or settings.properties_table_path
        logger.info("Initializing LocalDataProvider at base path: %s", self.base_path)

    def read_properties_full(self) -> DataFrame:
        """Reads properties_full from local filesystem."""
        gold_path = os.path.join(self.base_path, "gold", self.table_path)
        direct_path = os.path.join(self.base_path, self.table_path)

        file_path = gold_path if os.path.exists(gold_path) else direct_path
        logger.info("Reading properties_full from local storage: %s", file_path)
        try:
            df = self.spark.read.format("parquet").load(file_path)
            logger.info("Properties successfully loaded from local storage.")
            return df
        except Exception as e:
            logger.error("Failed to read properties_full from local storage (%s): %s", file_path, e)
            raise
