import json
from datetime import datetime

import numpy as np
import pandas as pd
import logging
from pyspark.sql import SparkSession, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from dataproviders import LocalDataProvider, DataProvider
from modelhandler import LocalModelHandler, ModelHandler, AzureModelHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize data provider and model handler
data_provider: DataProvider = LocalDataProvider()
model_handler: ModelHandler = LocalModelHandler("linear_regression")

def clean_data(properties_df: DataFrame) -> pd.DataFrame:
    """Cleans the input Spark DataFrame and returns a pandas DataFrame."""
    logger.info("Cleaning data...")
    clean_df = (
        properties_df
        .filter("type = 'apartment'")
        .filter("operation = 'buy'")
        .select("surface", "rooms", "bathrooms", "price", "elevator", "terrace", "garage", "postal_code")
        .dropna()
        .filter("surface > 0 AND rooms > 0 AND bathrooms > 0 AND price > 0")
    )
    logger.info(f"Cleaned dataset size: {clean_df.count()} rows")
    return clean_df.toPandas()

def generate_report(target_test: pd.Series, y_pred: np.ndarray, model: LinearRegression, features: pd.DataFrame) -> dict:
    """Generates a performance report for the model."""
    logger.info("Generating report...")
    mae = mean_absolute_error(target_test, y_pred)
    mse = mean_squared_error(target_test, y_pred)
    rmse = np.sqrt(mse)

    report = {
        "mae": mae,
        "rmse": rmse,
        "intercept": model.intercept_,
        "features": {feature: coef for feature, coef in zip(features.columns, model.coef_)}
    }
    logger.info(f"Report generated: MAE={mae}, RMSE={rmse}")
    return report

if __name__ == '__main__':
    logger.info("Starting Spark session...")
    spark = SparkSession.builder.appName("tfg-models").getOrCreate()

    logger.info("Reading property snapshot...")
    properties_df = data_provider.read_properties_snapshot(datetime(2025, 1, 25))

    # Clean and preprocess data
    clean_df = clean_data(properties_df)

    logger.info("Converting postal codes to dummy variables...")
    properties = pd.get_dummies(clean_df, columns=["postal_code"], drop_first=True)

    # Separate features and target
    features = properties.drop(columns=["price"])
    target = properties["price"]

    logger.info("Splitting data into training and testing sets...")
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    logger.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(features_train, target_train)

    logger.info("Making predictions...")
    y_pred = model.predict(features_test)

    # Generate and print report
    report = generate_report(target_test, y_pred, model, features_train)
    logger.info("Model performance report:")
    logger.info(json.dumps(report))

    logger.info("Saving model and metadata...")
    model_handler.save_model(
        model,
        features_train.columns,
        report=report
    )
    logger.info("Model and metadata saved successfully.")