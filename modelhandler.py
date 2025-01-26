import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import joblib
from azure.storage.blob import BlobServiceClient

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelHandler(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def save_model(self, model, features, report: dict):
        pass

    @abstractmethod
    def load_model(self) -> (Any, Any):
        pass


class LocalModelHandler(ModelHandler):

    _BASE_PATH = "/home/javi/local_datalake"

    def save_model(self, model, features, report: dict):
        model_path = f"{self._BASE_PATH}/models/{self.model_name}"
        os.makedirs(model_path, exist_ok=True)

        logging.info(f"Saving model to {model_path}")
        try:
            joblib.dump(model, f"{model_path}/model.pkl")
            joblib.dump(features, f"{model_path}/features.pkl")
            with open(f"{model_path}/report.json", "w") as f:
                json.dump(report, f)
            logging.info("Model, features, and report successfully saved.")
        except Exception as e:
            logging.error(f"Error saving the model locally: {e}")
            raise

    def load_model(self) -> (Any, Any):
        model_path = f"{self._BASE_PATH}/models/{self.model_name}"
        logging.info(f"Loading model from {model_path}")
        try:
            model = joblib.load(f"{model_path}/model.pkl")
            features = joblib.load(f"{model_path}/features.pkl")
            logging.info("Model and features successfully loaded.")
            return model, features
        except Exception as e:
            logging.error(f"Error loading the model locally: {e}")
            raise


class AzureModelHandler(ModelHandler):

    def _upload_blob(self, local_path: str, remote_path: str):
        logging.info(f"Uploading {local_path} to Azure Blob Storage at {remote_path}")
        try:
            blob_service_client = BlobServiceClient.from_connection_string(os.environ["tfgbs_connection_string"])
            container_client = blob_service_client.get_container_client("datalake")
            blob_client = container_client.get_blob_client(remote_path)
            with open(local_path, "rb") as f:
                blob_client.upload_blob(data=f, overwrite=True)
            logging.info(f"Upload completed: {remote_path}")
        except Exception as e:
            logging.error(f"Error uploading file to Azure Blob Storage: {e}")
            raise

    def save_model(self, model, features, report: dict):
        try:
            with tempfile.NamedTemporaryFile(delete=True) as model_temp:
                joblib.dump(model, model_temp.name)
                self._upload_blob(model_temp.name, f"models/{self.model_name}/model.pkl")
            with tempfile.NamedTemporaryFile(delete=True) as features_temp:
                joblib.dump(features, features_temp.name)
                self._upload_blob(features_temp.name, f"models/{self.model_name}/features.pkl")
            with tempfile.NamedTemporaryFile(mode='w', delete=True) as report_temp:
                json.dump(report, report_temp)
                self._upload_blob(report_temp.name, f"models/{self.model_name}/report.json")
            logging.info("Model, features, and report successfully uploaded to Azure.")
        except Exception as e:
            logging.error(f"Error saving the model to Azure: {e}")
            raise

    def load_model(self) -> (Any, Any):
        logging.info(f"Loading model from Azure for {self.model_name}")
        try:
            blob_service_client = BlobServiceClient.from_connection_string(os.environ["tfgbs_connection_string"])
            container_client = blob_service_client.get_container_client("datalake")

            # Download and load the model
            with tempfile.NamedTemporaryFile(delete=True) as model_temp:
                blob_client = container_client.get_blob_client(f"models/{self.model_name}/model.pkl")
                with open(model_temp.name, "wb") as f:
                    f.write(blob_client.download_blob().readall())
                model = joblib.load(model_temp.name)

            # Download and load the features
            with tempfile.NamedTemporaryFile(delete=True) as features_temp:
                blob_client = container_client.get_blob_client(f"models/{self.model_name}/features.pkl")
                with open(features_temp.name, "wb") as f:
                    f.write(blob_client.download_blob().readall())
                features = joblib.load(features_temp.name)

            logging.info("Model and features successfully loaded from Azure.")
            return model, features
        except Exception as e:
            logging.error(f"Error loading the model from Azure: {e}")
            raise
