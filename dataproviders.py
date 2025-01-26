import logging
from abc import ABC, abstractmethod
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProvider(ABC):

    @abstractmethod
    def read_properties_snapshot(self, date: datetime) -> DataFrame:
        pass

class AzureDataProvider(DataProvider):
    _BASE_URL = "abfss://datalake@tfgbs.dfs.core.windows.net/"

    def __init__(self):
        logging.info("Initializing AzureDataProvider...")
        self.spark = SparkSession.builder.appName("tfg-models").getOrCreate()
        logging.info("SparkSession created for AzureDataProvider.")

    def read_properties_snapshot(self, date: datetime):
        file_path = f"{self._BASE_URL}/processed/wallapop_properties_snapshots/year={date.year}/month={date.month}/day={date.day}"
        logging.info(f"Reading properties snapshot from Azure Data Lake: {file_path}")
        try:
            df = self.spark.read.format("parquet").load(file_path)
            logging.info("Data successfully loaded from Azure Data Lake.")
            return df
        except Exception as e:
            logging.error(f"Failed to read data from Azure Data Lake: {e}")
            raise

class LocalDataProvider(DataProvider):
    _BASE_URL = "/home/javi/local_datalake/"

    def __init__(self):
        logging.info("Initializing LocalDataProvider...")
        self.spark = SparkSession.builder.appName("tfg-models").getOrCreate()
        logging.info("SparkSession created for LocalDataProvider.")

    def read_properties_snapshot(self, date: datetime):
        file_path = f"{self._BASE_URL}/processed/wallapop_properties_snapshots/year={date.year}/month={date.month}/day={date.day}"
        logging.info(f"Reading properties snapshot from local storage: {file_path}")
        try:
            df = self.spark.read.format("parquet").load(file_path)
            logging.info("Data successfully loaded from local storage.")
            return df
        except Exception as e:
            logging.error(f"Failed to read data from local storage: {e}")
            raise
