# TFG Models: Real Estate Valuation Pipeline & APIs

Machine Learning pipelines and real-time inference services for real estate appraisal and valuation based on Azure Data Lake Storage Gen2 (`gold.properties_full`).

## Version
**0.0.2** (See [CHANGELOG.md](file:///Users/javierolmo/IdeaProjects/tfg-models/CHANGELOG.md) for release history)

---

## Dual Architecture: Training vs. Inference

The repository is split into two specialized runtime environments to optimize deployment on **Azure Container Apps** (scale-to-zero, low cold start latency, small resource instances) and batch training workflows:

| Feature / Metric | Inference Image (`Dockerfile.inference`) | Training Image (`Dockerfile.train`) |
| :--- | :--- | :--- |
| **Primary Target** | Azure Container Apps (HTTP API / Scale-to-Zero) | Azure ML / ACA Jobs / Batch Pipeline |
| **Heavy Libraries** | Excludes `pyarrow` (minimal runtime) | Includes `pyarrow`, parquet engines |
| **Cold-Start Time** | Ultra-fast (lightweight image layers) | Batch job startup (not latency-critical) |
| **Memory Footprint**| Low (~50-80 MB baseline, runs in 0.5 GiB RAM) | Standard (~200-500 MB RAM for training) |
| **Default Service** | FastAPI Inference API on port `8000` | FastAPI Training API on port `8001` / CLI |

---

## Project Structure

This project adheres to the standard Python `src/` layout:

```text
tfg-models/
├── .github/workflows/          # Automated CI/CD pipelines
│   ├── ci.yml                  # PR testing & verification for both Docker images
│   └── cd.yml                  # Auto-tagging & publishing inference/train images
├── CHANGELOG.md                # Release history and semantic version tracking
├── Dockerfile                  # Unified multi-stage build (targets: inference, train)
├── Dockerfile.inference        # Minimal standalone inference Dockerfile
├── Dockerfile.train            # Complete standalone training Dockerfile
├── docker-compose.yml          # Container orchestration (APIs & CLI services)
├── pyproject.toml              # Dependencies grouped into [inference], [train], [dev]
├── requirements-inference.txt  # Pinned inference dependencies
├── requirements-train.txt      # Pinned training dependencies
├── src/
│   └── tfg_models/             # Core package code
│       ├── __init__.py         # Package entry point and exports
│       ├── __main__.py         # Allows `python -m tfg_models`
│       ├── api/                # HTTP REST APIs (FastAPI)
│       │   ├── inference.py    # Real-time valuation API (port 8000)
│       │   ├── training.py     # Pipeline management & metrics API (port 8001)
│       │   └── schemas.py      # Pydantic schemas for requests/responses
│       ├── cli.py              # Unified CLI interface
│       ├── config.py           # Configuration and environment bindings
│       ├── core/               # Base trainer, AzureModelHandler, LocalModelHandler
│       ├── data/               # Cleaning and Data Lake providers (PyArrow)
│       └── models/             # Regressors (Linear Regression, LightGBM)
└── tests/                      # Automated unit and API test suites (43 tests, 88% coverage)
```

---

## HTTP REST APIs

### 1. Inference API (Port 8000)
Optimized for low-latency scoring and health checks in Azure Container Apps.

* `GET /health`: Readiness and liveness probe for Azure Container Apps.
* `GET /models`: Lists available and cached models.
* `GET /models/{model_name}/versions`: Lists saved versions in storage for a model.
* `POST /predict`: Calculates property valuation.

**Example Request (`POST /predict`):**
```json
{
  "model": "lightgbm",
  "version": "latest",
  "surface": 95.0,
  "rooms": 3,
  "bathrooms": 2,
  "postal_code": "36211",
  "elevator": true,
  "terrace": true,
  "garage": false
}
```

**Example Response:**
```json
{
  "model": "lightgbm",
  "version": "latest",
  "estimated_price": 245000.0,
  "currency": "EUR",
  "inputs": { ... }
}
```

### 2. Training API (Port 8001)
Used for pipeline orchestration, experiment tracking, and model comparison.

* `GET /health`: Health status of the training service.
* `POST /train`: Triggers model training (supports synchronous runs or asynchronous background jobs).
* `GET /train/jobs/{job_id}`: Inspects background training status and metrics.
* `GET /compare`: Returns benchmark metrics (MAE, RMSE, R²) for registered models.
* `GET /models/{model_name}/report`: Retrieves full evaluation report.

---

## Quickstart

### 1. Local Installation

```bash
# For inference only (minimal):
pip install -e ".[inference]"

# For training & full pipelines:
pip install -e ".[train]"

# For development and tests:
pip install -e ".[dev]"
```

### 2. Running the HTTP APIs

```bash
# Start Inference API
tfg-models-inference
# or: uvicorn tfg_models.api.inference:app --port 8000

# Start Training API
tfg-models-training
# or: uvicorn tfg_models.api.training:app --port 8001
```

### 3. Running via Docker Compose

```bash
# Launch Inference HTTP API (port 8000)
docker compose up inference-api

# Launch Training HTTP API (port 8001)
docker compose up training-api

# Run CLI batch training
docker compose run --rm train

# Run CLI prediction
docker compose run --rm predict
```

### 4. Building Docker Images

Using dedicated Dockerfiles:
```bash
# Build minimal inference image for Azure Container Apps
docker build -f Dockerfile.inference -t tfg-models-inference:latest .

# Build training image
docker build -f Dockerfile.train -t tfg-models-train:latest .
```

Using multi-stage targets:
```bash
docker build --target inference -t tfg-models-inference:latest .
docker build --target train -t tfg-models-train:latest .
```

### 5. Running Tests

```bash
pytest tests/ --cov=tfg_models --cov-report=term-missing
```

---

## CI/CD Workflows

### Continuous Integration (`ci.yml`)
* Runs unit test suite (`pytest`) with code coverage report.
* Validates build for both `Dockerfile.inference` and `Dockerfile.train`.

### Continuous Deployment (`cd.yml`)
* Triggers on merge to `main`.
* Automatically tags version `v<version>`.
* Builds multi-architecture images (`linux/amd64`, `linux/arm64`) and publishes to Docker Hub:
  * `<repo>-inference:latest`, `<repo>-inference:v<version>`
  * `<repo>-train:latest`, `<repo>-train:v<version>`
  * `<repo>:latest` (points to minimal inference image)
