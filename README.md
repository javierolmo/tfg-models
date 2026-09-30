# TFG Models: Real Estate Valuation Pipeline

Machine Learning pipelines for real estate appraisal and valuation based on Azure Data Lake Storage Gen2 (`gold.properties_full`).

## Version
**0.0.1**

## Architecture

This project follows the standard Python `src/` layout:

```text
tfg-models/
├── pyproject.toml              # Project specification, version (0.0.1), dependencies, console scripts
├── Dockerfile                  # Containerized ML environment (Python 3.12, Java 17, LightGBM)
├── docker-compose.yml          # Container orchestration (train, compare, predict)
├── src/
│   └── tfg_models/             # Core package
│       ├── __init__.py         # Package entry point and __version__
│       ├── cli.py              # Unified CLI interface
│       ├── config.py           # Typed configuration and environment settings
│       ├── core/               # Base abstractions (Template Method, Model Handlers)
│       ├── data/               # Data cleaning and Data Lake providers
│       └── models/             # Implementations (Linear Regression, LightGBM)
├── main.py                     # Convenience entry point
└── tests/                      # Automated unit and integration tests
```

## Quickstart

### 1. Installation

Install in editable mode:
```bash
pip install -e .
```

### 2. CLI Usage

You can use either `tfg-models` (installed CLI) or `python main.py`:

```bash
# Compare saved models performance
tfg-models compare

# Predict property valuation
tfg-models predict --model lightgbm --surface 95 --rooms 3 --bathrooms 2 --postal-code 36211 --elevator --terrace

# Train models on Azure Data Lake
tfg-models train --model all
```

### 3. Docker Execution

```bash
# Compare models
docker compose run --rm compare

# Predict
docker compose run --rm predict
```
