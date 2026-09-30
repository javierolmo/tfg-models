"""Unit tests for data preprocessing module."""

import pandas as pd

from tfg_models.data.preprocessing import clean_property_data, prepare_features_for_model


def test_clean_property_data():
    raw_df = pd.DataFrame(
        [
            # Valid Flat, Sell
            {"property_id": 1, "type": "Flat", "operation": "Sell", "surface": 80.0, "rooms": 3, "bathrooms": 1, "price": 200000.0, "postal_code": "36201", "elevator": None, "terrace": True, "garage": None},
            # Valid apartment, buy
            {"property_id": 2, "type": "apartment", "operation": "buy", "surface": 95.0, "rooms": 4, "bathrooms": 2, "price": 310000.0, "postal_code": "36211", "elevator": True, "terrace": False, "garage": True},
            # Invalid type: House
            {"property_id": 3, "type": "House", "operation": "Sell", "surface": 150.0, "rooms": 5, "bathrooms": 3, "price": 500000.0, "postal_code": "36201", "elevator": False, "terrace": True, "garage": True},
            # Invalid operation: Rent
            {"property_id": 4, "type": "Flat", "operation": "Rent", "surface": 60.0, "rooms": 2, "bathrooms": 1, "price": 800.0, "postal_code": "36201", "elevator": True, "terrace": False, "garage": False},
            # Invalid negative surface
            {"property_id": 5, "type": "Flat", "operation": "Sell", "surface": -10.0, "rooms": 2, "bathrooms": 1, "price": 100000.0, "postal_code": "36201", "elevator": True, "terrace": False, "garage": False},
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
