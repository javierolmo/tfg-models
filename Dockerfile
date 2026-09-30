FROM python:3.12-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOCAL_DATALAKE_PATH=/app/local_datalake

# Install required system dependencies on Debian Bookworm:
# - openjdk-17-jre-headless: Java 17 LTS runtime for PySpark
# - libgomp1: OpenMP runtime required by LightGBM
# - procps & curl: Process and networking utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    libgomp1 \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure Java environment for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

# Install Python dependencies first for caching efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Install package in editable mode so 'tfg-models' CLI is globally available in PATH
RUN pip install --no-cache-dir -e .

# Directory for mounting local models or datalake volume
RUN mkdir -p /app/local_datalake

# Default entrypoint delegates to unified CLI
ENTRYPOINT ["tfg-models"]
CMD ["--help"]
