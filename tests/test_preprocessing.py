"""Unit tests for data preprocessing module."""

import numpy as np
import pandas as pd

from tfg_models.data.preprocessing import (
    clean_property_data,
    prepare_features_for_model,
    prepare_inference_features,
)


def test_clean_property_data():
    raw_df = pd.DataFrame(
        [
            # Valid Flat, Sell
            {"property_id": 1, "type": "FLAT", "operation": "SELL", "surface": 80.0, "rooms": 3, "bathrooms": 1, "price": 200000.0, "postal_code": "36201", "elevator": None, "terrace": True, "garage": None},
            # Valid Flat, Sell
            {"property_id": 2, "type": "flat", "operation": "sell", "surface": 95.0, "rooms": 4, "bathrooms": 2, "price": 310000.0, "postal_code": "36211", "elevator": True, "terrace": False, "garage": True},
            # Invalid type: House
            {"property_id": 3, "type": "HOUSE", "operation": "SELL", "surface": 150.0, "rooms": 5, "bathrooms": 3, "price": 500000.0, "postal_code": "36201", "elevator": False, "terrace": True, "garage": True},
            # Invalid operation: Rent
            {"property_id": 4, "type": "FLAT", "operation": "RENT", "surface": 60.0, "rooms": 2, "bathrooms": 1, "price": 800.0, "postal_code": "36201", "elevator": True, "terrace": False, "garage": False},
            # Invalid negative surface
            {"property_id": 5, "type": "FLAT", "operation": "SELL", "surface": -10.0, "rooms": 2, "bathrooms": 1, "price": 100000.0, "postal_code": "36201", "elevator": True, "terrace": False, "garage": False},
        ]
    )

    cleaned = clean_property_data(raw_df)

    # Only first two rows are valid
    assert len(cleaned) == 2

    # Check null imputation
    row1 = cleaned[cleaned["surface"] == 80.0].iloc[0]
    assert row1["elevator"] == 0
    assert row1["terrace"] == 1
    assert row1["garage"] == 0

    row2 = cleaned[cleaned["surface"] == 95.0].iloc[0]
    assert row2["elevator"] == 1
    assert row2["terrace"] == 0
    assert row2["garage"] == 1


def test_clean_property_data_with_gold_parquet_schema():
    """Validates behavior against newly updated Gold PropertiesFull types (Integer postal_code, uppercase types)."""
    gold_df = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            # Uppercase standardized types from wallascala PropertyTypeStandardizer & OperationStandardizer
            "type": ["FLAT", "FLAT", "HOUSE", "FLAT"],
            "operation": ["SELL", "SELL", "SELL", "RENT"],
            "surface": [70, 110, 200, 50],
            "rooms": [2, 4, 5, 1],
            "bathrooms": [1, 2, 3, 1],
            "price": [180000, 350000, 500000, 750],
            # Spark IntegerType postal_code with leading zero dropped (8001 for Barcelona 08001)
            "postal_code": [8001.0, 36211.0, 28001.0, np.nan],
            # Nullable booleans
            "elevator": pd.Series([True, None, False, True], dtype="boolean"),
            "terrace": pd.Series([None, True, False, False], dtype="boolean"),
            "garage": pd.Series([False, True, None, False], dtype="boolean"),
        }
    )

    cleaned = clean_property_data(gold_df)

    # Only rows 1 and 2 are valid FLAT + SELL with valid postal code
    assert len(cleaned) == 2

    # Verify postal codes were padded to 5-digit strings
    assert list(cleaned["postal_code"]) == ["08001", "36211"]

    # Verify booleans correctly converted to ints
    assert list(cleaned["elevator"]) == [1, 0]
    assert list(cleaned["terrace"]) == [0, 1]
    assert list(cleaned["garage"]) == [0, 1]


def test_prepare_features_for_model_onehot():
    df = pd.DataFrame(
        {
            "surface": [50.0, 75.0, 100.0],
            "rooms": [2, 3, 4],
            "bathrooms": [1, 2, 2],
            "elevator": [False, True, True],
            "terrace": [False, False, True],
            "garage": [False, True, True],
            "postal_code": ["36201", "36202", "36201"],
            "price": [120000.0, 190000.0, 260000.0],
        }
    )

    X, y = prepare_features_for_model(df, encoding="onehot")

    assert len(y) == 3
    assert list(y) == [120000.0, 190000.0, 260000.0]
    assert "surface" in X.columns
    assert "postal_code_36202" in X.columns
    assert "price" not in X.columns


def test_prepare_features_for_model_categorical():
    df = pd.DataFrame(
        {
            "surface": [50.0, 75.0],
            "rooms": [2, 3],
            "bathrooms": [1, 2],
            "elevator": [False, True],
            "terrace": [False, False],
            "garage": [False, True],
            "postal_code": ["36201", "36202"],
            "price": [120000.0, 190000.0],
        }
    )

    X, y = prepare_features_for_model(df, encoding="categorical")

    assert X["postal_code"].dtype.name == "category"
    assert "postal_code" in X.columns
    assert "postal_code_36201" not in X.columns


def test_prepare_inference_features_categorical():
    features = ["surface", "rooms", "bathrooms", "elevator", "terrace", "garage", "postal_code"]
    df = prepare_inference_features(
        surface=85.0,
        rooms=3,
        bathrooms=2,
        postal_code="8001",
        elevator=True,
        terrace=False,
        garage=True,
        features=features,
        encoding="categorical",
    )
    assert len(df) == 1
    assert df["surface"].iloc[0] == 85.0
    assert df["postal_code"].iloc[0] == "08001"
    assert df["postal_code"].dtype.name == "category"
    assert df["elevator"].iloc[0] == 1
    assert df["terrace"].iloc[0] == 0
    assert df["garage"].iloc[0] == 1


def test_prepare_inference_features_onehot():
    features = ["surface", "rooms", "bathrooms", "elevator", "terrace", "garage", "postal_code_36211"]
    df = prepare_inference_features(
        surface=90.0,
        rooms=3,
        bathrooms=1,
        postal_code=36211,
        elevator=False,
        terrace=True,
        garage=False,
        features=features,
        encoding="onehot",
    )
    assert len(df) == 1
    assert bool(df["postal_code_36211"].iloc[0]) is True
    assert list(df.columns) == features
