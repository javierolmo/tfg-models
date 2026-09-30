# TFG Models: Real Estate Valuation Pipeline

Machine Learning pipelines for real estate appraisal and valuation based on Azure Data Lake Storage Gen2 (`gold.properties_full`).

## Version
**0.0.1** (See [CHANGELOG.md](file:///Users/javierolmo/IdeaProjects/tfg-models/CHANGELOG.md) for release history)

## Project Structure

This project adheres to the standard Python `src/` layout:

```text
tfg-models/
├── CHANGELOG.md                # Release history and semantic version tracking
├── pyproject.toml              # Project specification, version (0.0.1), dependencies, console scripts
├── Dockerfile                  # Containerized ML environment (Python 3.12, Java 17, LightGBM)
├── docker-compose.yml          # Container orchestration (train, compare, predict)
├── src/
│   └── tfg_models/             # 100% of the project package code
│       ├── __init__.py         # Package entry point and __version__
│       ├── __main__.py         # Allows `python -m tfg_models`
│       ├── cli.py              # Unified CLI interface
│       ├── config.py           # Typed configuration and environment settings
│       ├── core/               # Base abstractions (Template Method, Azure Model Handler)
│       ├── data/               # Data cleaning and Data Lake providers
│       └── models/             # Implementations (Linear Regression, LightGBM)
└── tests/                      # Automated unit tests and test fixtures
    ├── helpers/                # Test-only fixtures (LocalModelHandler)
    └── test_*.py               # Test suites (89% coverage)
```

## Quickstart

### 1. Installation

Install in editable mode:
```bash
pip install -e .
```

### 2. Execution

You can run commands using the installed CLI `tfg-models` or via `python -m tfg_models`:

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

# Predict valuation
docker compose run --rm predict
```

### 4. Running Tests

```bash
pytest tests/ --cov=tfg_models --cov-report=term-missing
```
