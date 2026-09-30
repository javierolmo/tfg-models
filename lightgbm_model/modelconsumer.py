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

from modelhandler import AzureModelHandler, LocalModelHandler, ModelHandler

MODEL_HANDLER_TYPE = os.environ.get("MODEL_HANDLER", "azure").lower()
if MODEL_HANDLER_TYPE == "azure":
    model_handler: ModelHandler = AzureModelHandler("lightgbm")
else:
    model_handler: ModelHandler = LocalModelHandler("lightgbm")


def predict(
    model: lgb.LGBMRegressor,
    features: list,
    surface: int,
    rooms: int,
    bathrooms: int,
    elevator: bool,
    terrace: bool,
    garage: bool,
    postal_code: int,
) -> np.ndarray:
    """Runs valuation inference using the trained LightGBM model."""
    new_data = {
        "surface": [surface],
        "rooms": [rooms],
        "bathrooms": [bathrooms],
        "elevator": [int(elevator)],
        "terrace": [int(terrace)],
        "garage": [int(garage)],
        "postal_code": [postal_code],
    }

    new_df = pd.DataFrame(new_data)

    # Cast postal_code to category matching the training preprocessing
    new_df["postal_code"] = new_df["postal_code"].astype("category")

    # Align columns to match training features exactly
    new_df = new_df[features]

    predictions = model.predict(new_df)
    print("Predictions:", predictions)
    return predictions


if __name__ == '__main__':
    model, features = model_handler.load_model()
    predict(
        model=model,
        features=features,
        surface=80,
        rooms=3,
        bathrooms=1,
        elevator=True,
        terrace=False,
        garage=False,
        postal_code=36211,
    )
