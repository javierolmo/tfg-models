# ==============================================================================
# Base Layer: Python 3.12 slim with OpenMP runtime (libgomp1) for LightGBM
# ==============================================================================
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

# ==============================================================================
# Target: inference (Minimal image for Azure Container Apps, Scale-to-Zero)
# ==============================================================================
FROM base AS inference

ENV PORT=8000

# Install only inference dependencies (fastapi, uvicorn, pydantic - excludes pyarrow)
RUN pip install --no-cache-dir ".[inference]"

EXPOSE 8000

ENTRYPOINT ["uvicorn", "tfg_models.api.inference:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Target: train (Full image for training pipelines, PyArrow, Azure Data Lake)
# ==============================================================================
FROM base AS train

ENV LOCAL_DATALAKE_PATH=/app/local_datalake \
    PORT=8001

# Install full dependencies including pyarrow
RUN pip install --no-cache-dir ".[train]" \
    && mkdir -p /app/local_datalake

EXPOSE 8001

ENTRYPOINT ["uvicorn", "tfg_models.api.training:app"]
CMD ["--host", "0.0.0.0", "--port", "8001"]
