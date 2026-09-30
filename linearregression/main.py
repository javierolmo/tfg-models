"""Linear regression training entry point."""

import sys
from pathlib import Path

SRC_ROOT = str(Path(__file__).resolve().parent.parent / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from tfg_models.models.linear_regression import LinearRegressionTrainer

if __name__ == '__main__':
    trainer = LinearRegressionTrainer()
    trainer.run()
