FROM python:3.12-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOCAL_DATALAKE_PATH=/app/local_datalake

# Install minimal runtime dependencies:
# - libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package and production dependencies cleanly
RUN pip install --no-cache-dir .

RUN mkdir -p /app/local_datalake

# Default entrypoint delegates to unified CLI
ENTRYPOINT ["tfg-models"]
CMD ["--help"]
