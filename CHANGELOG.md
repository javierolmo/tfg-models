# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-09-30

### Added
- **Standardized `src/` layout**: Migrated package source tree to canonical `src/tfg_models/` structure.
- **Project descriptor**: Added `pyproject.toml` (PEP 518/621) with metadata, dependencies, and `tfg-models` console script entrypoint.
- **Runnable module**: Added `src/tfg_models/__main__.py` allowing execution via `python -m tfg_models`.
- **LightGBM Regressor**: Added `LightGBMTrainer` with native categorical support for postal codes, achieving ~15% MAE reduction over linear baseline.
- **Unified CLI**: CLI interface (`train`, `predict`, `compare`, `versions`).
- **Template Method pattern**: Implemented `BaseModelTrainer` lifecycle (`load -> preprocess -> split -> fit -> evaluate -> save`).
- **MLOps versioning**: Integrated timestamped versions (`YYYYMMDD_HHMMSS`), git commit hash tracking, and `latest` pointer in model persistence.
- **Dockerization**:
  - Multi-platform `Dockerfile` with Debian Bookworm, Python 3.12, OpenJDK 17, and OpenMP `libgomp1`.
  - `docker-compose.yml` defining `train`, `compare`, and `predict` services.
- **Automated CI/CD Workflows**:
  - `CI` (`ci.yml`): Runs unit tests and coverage across PRs targeting `main` with Java 17 and OpenMP setup.
  - `CD` (`cd.yml`): Automatically generates Git release tags from `pyproject.toml` on merge to `main`, builds multi-arch Docker images, and pushes to Docker Hub (`v<version>`, `<version>`, and `latest`).
- **Comprehensive test suite**:
  - 28 unit tests covering `cli`, `config`, `model_handler`, `models`, `preprocessing`, `providers`, and `trainer`.
  - 89% code coverage with `pytest-cov`.
  - Isolated test fixtures and `LocalModelHandler` in `tests/helpers/`.

### Changed
- Production persistence defaults strictly to `AzureModelHandler` (Azure Blob Storage).
- Consolidated data preprocessing and feature extraction into `data/preprocessing.py`.
- Optimized inference performance to bypass SparkSession initialization during prediction.

### Removed
- Legacy facade files and obsolete root directories (`dataproviders.py`, `modelhandler.py`, `linearregression/`, `lightgbm_model/`, `main.py`).
- Removed `LocalModelHandler` from production package (`src/tfg_models/core/model_handler.py`).
