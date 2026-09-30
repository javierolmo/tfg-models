"""Basic package tests."""

import tfg_models


def test_version():
    assert tfg_models.__version__ == "0.0.1"


def test_registry():
    assert "linear_regression" in tfg_models.models.MODEL_REGISTRY
    assert "lightgbm" in tfg_models.models.MODEL_REGISTRY
