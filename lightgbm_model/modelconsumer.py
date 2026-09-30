"""LightGBM model consumer / prediction entry point."""

import sys
from pathlib import Path

SRC_ROOT = str(Path(__file__).resolve().parent.parent / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from tfg_models.models.lightgbm_model import LightGBMTrainer

if __name__ == '__main__':
    trainer = LightGBMTrainer()
    price = trainer.predict_sample(
        surface=80,
        rooms=3,
        bathrooms=1,
        postal_code="36211",
        elevator=False,
        terrace=False,
        garage=False,
    )
    print("Predictions:", [price])
