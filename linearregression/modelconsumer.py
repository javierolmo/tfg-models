"""Linear regression inference consumer."""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.linear_regression import LinearRegressionTrainer


def predict(surface: int, rooms: int, bathrooms: int, postal_code: int, **kwargs):
    trainer = LinearRegressionTrainer()
    return trainer.predict_sample(
        surface=surface,
        rooms=rooms,
        bathrooms=bathrooms,
        postal_code=postal_code,
        **kwargs,
    )


if __name__ == '__main__':
    price = predict(
        surface=80,
        rooms=3,
        bathrooms=1,
        elevator=True,
        terrace=False,
        garage=False,
        postal_code=36211,
    )
    print(f"Predictions: [{price}]")
