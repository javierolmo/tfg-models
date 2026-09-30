"""Linear regression training entry point."""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.linear_regression import LinearRegressionTrainer

if __name__ == '__main__':
    trainer = LinearRegressionTrainer()
    trainer.run()
