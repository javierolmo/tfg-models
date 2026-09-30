"""LightGBM training entry point."""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.lightgbm_model import LightGBMTrainer

if __name__ == '__main__':
    trainer = LightGBMTrainer()
    trainer.run()
