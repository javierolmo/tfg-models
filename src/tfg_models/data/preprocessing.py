"""Common data preprocessing and cleaning functions for property datasets."""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def clean_property_data(properties_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw property DataFrame using vectorized Pandas operations:
    - Filters by apartment/flat type and sell/buy operations.
    - Imputes boolean flags (elevator, terrace, garage) with 0 if NULL/False.
    - Drops records with nulls in critical predictors (surface, rooms, bathrooms, postal_code, price).
    - Ensures positive physical measurements and price.
    - Standardizes data types.
    """
    logger.info("Cleaning property dataset...")
    df = properties_df.copy()

    # Filter by property type and operation (case-insensitive)
    if "type" in df.columns:
        type_mask = df["type"].astype(str).str.lower().isin(["flat", "apartment"])
        df = df[type_mask]

    if "operation" in df.columns:
        op_mask = df["operation"].astype(str).str.lower().isin(["sell", "buy"])
        df = df[op_mask]

    # Select relevant columns if present
    target_columns = ["surface", "rooms", "bathrooms", "price", "elevator", "terrace", "garage", "postal_code"]
    available_cols = [c for c in target_columns if c in df.columns]
    df = df[available_cols]

    # Impute boolean flags with 0 if null
    for bool_col in ["elevator", "terrace", "garage"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].fillna(0).astype(int)

    # Drop nulls in critical predictors and target
    critical_cols = [c for c in ["surface", "rooms", "bathrooms", "price", "postal_code"] if c in df.columns]
    df = df.dropna(subset=critical_cols)

    # Ensure numeric types
    for num_col in ["surface", "rooms", "bathrooms", "price"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Drop any rows where coercion created NaNs
    df = df.dropna(subset=[c for c in ["surface", "rooms", "bathrooms", "price"] if c in df.columns])

    # Filter positive physical measurements and price
    positive_mask = (
        (df["surface"] > 0)
        & (df["rooms"] > 0)
        & (df["bathrooms"] > 0)
        & (df["price"] > 0)
    )
    df = df[positive_mask]

    logger.info("Cleaned dataset size: %d rows", len(df))
    return df


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
