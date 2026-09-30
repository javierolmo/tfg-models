"""Common data preprocessing and cleaning functions for property datasets."""

import logging
from typing import Tuple

import pandas as pd
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def clean_property_data(properties_df: DataFrame) -> pd.DataFrame:
    """
    Cleans raw property Spark DataFrame:
    - Filters by apartment/flat type and sell/buy operations.
    - Imputes boolean flags (elevator, terrace, garage) with False if NULL.
    - Drops records with nulls in critical predictors (surface, rooms, bathrooms, postal_code, price).
    - Ensures positive physical measurements and price.
    - Converts to Pandas DataFrame and standardizes data types.
    """
    logger.info("Cleaning property dataset...")
    clean_spark_df = (
        properties_df
        .filter("lower(type) in ('flat', 'apartment')")
        .filter("lower(operation) in ('sell', 'buy')")
        .select("surface", "rooms", "bathrooms", "price", "elevator", "terrace", "garage", "postal_code")
        .fillna({"elevator": False, "terrace": False, "garage": False})
        .dropna(subset=["surface", "rooms", "bathrooms", "price", "postal_code"])
        .filter("surface > 0 AND rooms > 0 AND bathrooms > 0 AND price > 0")
    )

    count = clean_spark_df.count()
    logger.info("Cleaned dataset size: %d rows", count)

    pdf = clean_spark_df.toPandas()

    # Convert booleans to binary integer indicators (0 / 1)
    for col in ["elevator", "terrace", "garage"]:
        if col in pdf.columns:
            pdf[col] = pdf[col].astype(int)

    # Ensure numeric columns are properly typed
    for col in ["surface", "rooms", "bathrooms", "price"]:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    return pdf


def prepare_features_for_model(
    df: pd.DataFrame,
    encoding: str = "categorical",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepares features (X) and target (y) from a cleaned DataFrame.

    Args:
        df: Cleaned pandas DataFrame containing 'price' and predictor columns.
        encoding: 'categorical' for native category handling (e.g. LightGBM, CatBoost)
                  or 'onehot' for dummy variable encoding (e.g. Linear Regression).

    Returns:
        Tuple of (features_df, target_series).
    """
    df_copy = df.copy()

    if encoding == "onehot":
        logger.info("Encoding postal_code with one-hot dummy variables...")
        df_encoded = pd.get_dummies(df_copy, columns=["postal_code"], drop_first=True)
        features = df_encoded.drop(columns=["price"])
        target = df_encoded["price"]
    elif encoding == "categorical":
        logger.info("Setting postal_code to category dtype for native tree splitting...")
        df_copy["postal_code"] = df_copy["postal_code"].astype("category")
        features = df_copy.drop(columns=["price"])
        target = df_copy["price"]
    else:
        features = df_copy.drop(columns=["price"])
        target = df_copy["price"]

    return features, target
