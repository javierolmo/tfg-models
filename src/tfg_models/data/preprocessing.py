"""Common data preprocessing and cleaning functions for property datasets."""

import logging
from typing import Any, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def clean_property_data(properties_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw property DataFrame using vectorized Pandas operations:
    - Filters strictly by Gold layer contract: type 'FLAT' and operation 'SELL'.
    - Imputes boolean flags (elevator, terrace, garage) with 0 if NULL/False.
    - Standardizes postal codes to 5-digit zero-padded strings (e.g. 8001 -> '08001', 36211.0 -> '36211').
    - Drops records with nulls or non-numeric values in critical predictors.
    - Ensures positive physical measurements and price.
    - Standardizes data types.
    """
    logger.info("Cleaning property dataset...")
    df = properties_df.copy()

    # Filter strictly by Gold standard contract: type FLAT and operation SELL
    if "type" in df.columns:
        df = df[df["type"].astype(str).str.upper() == "FLAT"]

    if "operation" in df.columns:
        df = df[df["operation"].astype(str).str.upper() == "SELL"]

    # Select relevant columns if present
    target_columns = ["surface", "rooms", "bathrooms", "price", "elevator", "terrace", "garage", "postal_code"]
    available_cols = [c for c in target_columns if c in df.columns]
    df = df[available_cols]

    # Impute boolean flags with 0 if null/False (safe against nullable boolean and object dtypes)
    for bool_col in ["elevator", "terrace", "garage"]:
        if bool_col in df.columns:
            df[bool_col] = (df[bool_col] == True).fillna(False).astype(int)

    # Standardize postal codes to 5-digit zero-padded strings if present
    # Handled formats: integer (8001 -> '08001'), float (36211.0 -> '36211'), string ('36211' / '08001')
    if "postal_code" in df.columns:
        num_pc = pd.to_numeric(df["postal_code"], errors="coerce")
        valid_pc = num_pc.notna() & (num_pc >= 1000) & (num_pc <= 99999)
        df["postal_code"] = None
        df.loc[valid_pc, "postal_code"] = num_pc[valid_pc].astype(int).astype(str).str.zfill(5)

    # Coerce numeric predictors and price
    for num_col in ["surface", "rooms", "bathrooms", "price"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # Drop nulls in critical predictors and target in a single pass
    critical_cols = [c for c in ["surface", "rooms", "bathrooms", "price", "postal_code"] if c in df.columns]
    df = df.dropna(subset=critical_cols)

    # Filter positive physical measurements and price
    positive_mask = (
        (df["surface"] > 0)
        & (df["rooms"] > 0)
        & (df["bathrooms"] > 0)
        & (df["price"] > 0)
    )
    df = df[positive_mask]

    # Ensure postal_code is strictly string type
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].astype(str)

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

    if "postal_code" in df_copy.columns:
        df_copy["postal_code"] = df_copy["postal_code"].astype(str)

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


def prepare_inference_features(
    surface: float,
    rooms: int,
    bathrooms: int,
    postal_code: Any,
    elevator: bool = False,
    terrace: bool = False,
    garage: bool = False,
    features: Optional[List[str]] = None,
    encoding: str = "categorical",
) -> pd.DataFrame:
    """
    Constructs and formats a single-sample feature DataFrame matching the model's training expectations.
    """
    # Standardize 5-digit postal code (e.g. 8001 -> '08001', '36211' -> '36211')
    standardized_pc = str(postal_code).split(".")[0].strip().zfill(5)

    input_data = pd.DataFrame(
        [
            {
                "surface": float(surface),
                "rooms": int(rooms),
                "bathrooms": int(bathrooms),
                "elevator": int(bool(elevator)),
                "terrace": int(bool(terrace)),
                "garage": int(bool(garage)),
                "postal_code": standardized_pc,
            }
        ]
    )

    if encoding == "onehot":
        if features:
            for col in features:
                if col not in input_data.columns:
                    input_data[col] = False
            target_pc_col = f"postal_code_{standardized_pc}"
            if target_pc_col in input_data.columns:
                input_data[target_pc_col] = True
            return input_data[features]
        return input_data
    else:
        input_data["postal_code"] = input_data["postal_code"].astype("category")
        if features:
            return input_data[features]
        return input_data
