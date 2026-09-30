"""Unit tests for data preprocessing module."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from tfg_models.data.preprocessing import clean_property_data, prepare_features_for_model


@pytest.fixture(scope="module")
def spark_session():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("test-preprocessing")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def test_clean_property_data(spark_session):
    raw_data = [
        # Valid Flat, Sell
        (1, "Flat", "Sell", 80.0, 3, 1, 200000.0, "36201", None, True, None),
        # Valid apartment, buy
        (2, "apartment", "buy", 95.0, 4, 2, 310000.0, "36211", True, False, True),
        # Invalid type: House
        (3, "House", "Sell", 150.0, 5, 3, 500000.0, "36201", False, True, True),
        # Invalid operation: Rent
        (4, "Flat", "Rent", 60.0, 2, 1, 800.0, "36201", True, False, False),
    ]

    schema = [
        "property_id",
        "type",
        "operation",
        "surface",
        "rooms",
        "bathrooms",
        "price",
        "postal_code",
        "elevator",
        "terrace",
        "garage",
    ]

    spark_df = spark_session.createDataFrame(raw_data, schema)
    cleaned = clean_property_data(spark_df)

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
