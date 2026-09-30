import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path so modules can be imported when running script directly
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import lightgbm as lgb
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from dataproviders import AzureDataProvider, DataProvider, LocalDataProvider
from modelhandler import AzureModelHandler, LocalModelHandler, ModelHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine provider and handler based on environment (defaults to Azure for data, local for model saving)
DATA_PROVIDER_TYPE = os.environ.get("DATA_PROVIDER", "azure").lower()
MODEL_HANDLER_TYPE = os.environ.get("MODEL_HANDLER", "local").lower()

if DATA_PROVIDER_TYPE == "azure":
    data_provider: DataProvider = AzureDataProvider()
else:
    data_provider: DataProvider = LocalDataProvider()

if MODEL_HANDLER_TYPE == "azure":
    model_handler: ModelHandler = AzureModelHandler("lightgbm")
else:
    model_handler: ModelHandler = LocalModelHandler("lightgbm")


def clean_data(properties_df: DataFrame) -> pd.DataFrame:
    """Cleans the input Spark DataFrame and returns a pandas DataFrame prepared for LightGBM."""
    logger.info("Cleaning data...")
    clean_df = (
        properties_df
        .filter("lower(type) in ('flat', 'apartment')")
        .filter("lower(operation) in ('sell', 'buy')")
        .select("surface", "rooms", "bathrooms", "price", "elevator", "terrace", "garage", "postal_code")
        .fillna({"elevator": False, "terrace": False, "garage": False})
        .dropna(subset=["surface", "rooms", "bathrooms", "price", "postal_code"])
        .filter("surface > 0 AND rooms > 0 AND bathrooms > 0 AND price > 0")
    )
    logger.info(f"Cleaned dataset size: {clean_df.count()} rows")
    pdf = clean_df.toPandas()

    # Convert boolean columns to integer
    for col in ["elevator", "terrace", "garage"]:
        pdf[col] = pdf[col].astype(int)

    # Use category dtype for native high-cardinality categorical handling in LightGBM
    pdf["postal_code"] = pdf["postal_code"].astype("category")

    return pdf


def generate_report(
    target_test: pd.Series,
    y_pred: np.ndarray,
    model: lgb.LGBMRegressor,
    feature_names: list,
) -> dict:
    """Generates a performance report for the LightGBM model."""
    logger.info("Generating report...")
    mae = mean_absolute_error(target_test, y_pred)
    mse = mean_squared_error(target_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(target_test, y_pred)

    importances = model.feature_importances_
    feature_importance_dict = {
        name: int(imp) for name, imp in zip(feature_names, importances)
    }
    # Sort feature importances descending
    sorted_importances = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    report = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "feature_importances": sorted_importances,
        "n_estimators": int(model.n_estimators),
        "learning_rate": float(model.learning_rate),
        "num_leaves": int(model.num_leaves),
    }
    logger.info(f"Report generated: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    return report


if __name__ == '__main__':
    logger.info("Reading properties dataset (properties_full)...")
    properties_df = data_provider.read_properties_full()

    # Clean and preprocess data
    properties = clean_data(properties_df)

    # Separate features and target
    features = properties.drop(columns=["price"])
    target = properties["price"]

    logger.info("Splitting data into training and testing sets (size: %d)...", len(properties))
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    logger.info("Training LightGBM Regressor model...")
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        objective="regression",
        verbosity=-1,
    )
    model.fit(features_train, target_train)

    logger.info("Making predictions...")
    y_pred = model.predict(features_test)

    # Generate and log report
    report = generate_report(target_test, y_pred, model, list(features_train.columns))
    logger.info("Model performance summary:")
    logger.info("MAE: %.2f | RMSE: %.2f | R2: %.4f", report["mae"], report["rmse"], report["r2_score"])
    logger.info("Feature importances:\n%s", json.dumps(report["feature_importances"], indent=2))

    logger.info("Saving model and metadata...")
    model_handler.save_model(
        model,
        list(features_train.columns),
        report=report,
    )
    logger.info("LightGBM model and metadata saved successfully.")
