import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pyspark.sql import DataFrame, SparkSession

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

    DEFAULT_STORAGE_ACCOUNT = "tfgdatalake"
    DEFAULT_CONTAINER = "gold"
    DEFAULT_TABLE_PATH = "properties_full"

    def __init__(
        self,
        storage_account: Optional[str] = None,
        container: Optional[str] = None,
        table_path: Optional[str] = None,
        account_key: Optional[str] = None,
        spark_session: Optional[SparkSession] = None,
    ):
        self.storage_account = (
            storage_account
            or os.environ.get("AZURE_STORAGE_ACCOUNT", self.DEFAULT_STORAGE_ACCOUNT)
        )
        self.container = (
            container
            or os.environ.get("AZURE_CONTAINER", self.DEFAULT_CONTAINER)
        )
        self.table_path = (
            table_path
            or os.environ.get("PROPERTIES_PATH", self.DEFAULT_TABLE_PATH)
        )
        self.account_key = (
            account_key
            or os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            or os.environ.get("AZURE_STORAGE_KEY")
            or os.environ.get("TFGBS_KEY")
        )

        if not spark_session:
            builder = SparkSession.builder.appName("tfg-models")
            # Include Hadoop Azure connector jars
            builder = builder.config(
                "spark.jars.packages",
                "org.apache.hadoop:hadoop-azure:3.4.0,com.microsoft.azure:azure-storage:8.6.6",
            )
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
            # If SparkSession was already created, set Hadoop configuration directly
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

    DEFAULT_BASE_PATH = os.path.expanduser("~/local_datalake")
    DEFAULT_TABLE_PATH = "properties_full"

    def __init__(
        self,
        base_path: Optional[str] = None,
        table_path: Optional[str] = None,
        spark_session: Optional[SparkSession] = None,
    ):
        if not spark_session:
            spark_session = SparkSession.builder.appName("tfg-models").getOrCreate()

        super().__init__(spark_session=spark_session)
        self.base_path = base_path or os.environ.get("LOCAL_DATALAKE_PATH", self.DEFAULT_BASE_PATH)
        self.table_path = table_path or os.environ.get("PROPERTIES_PATH", self.DEFAULT_TABLE_PATH)
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
