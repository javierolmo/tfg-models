"""FastAPI Inference Service optimized for Azure Container Apps."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, status

from tfg_models import __version__
from tfg_models.api.schemas import HealthResponse, PredictRequest, PredictResponse
from tfg_models.core.model_handler import ModelHandler, get_model_handler
from tfg_models.data.preprocessing import prepare_inference_features
from tfg_models.models import MODEL_REGISTRY, get_model_trainer

logger = logging.getLogger(__name__)

# Global cache: (model_name, version) -> (model, features, encoding)
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, List[str], str]] = {}


def load_and_cache_model(model_name: str, version: str = "latest") -> Tuple[Any, List[str], str]:
    """Loads a model into cache if not present."""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown model architecture: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}",
        )

    cache_key = (model_name, version)
    if cache_key not in _MODEL_CACHE:
        handler: ModelHandler = get_model_handler(model_name)
        trainer = get_model_trainer(model_name)
        try:
            model, features = handler.load_model(version=version)
            _MODEL_CACHE[cache_key] = (model, features, trainer.encoding)
            logger.info("Loaded and cached model '%s' (version: %s)", model_name, version)
        except Exception as e:
            logger.error("Failed to load model '%s' (version: %s): %s", model_name, version, e)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not load model '{model_name}' (version '{version}'): {str(e)}",
            )
    return _MODEL_CACHE[cache_key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warmup model on container startup to minimize cold-start latency."""
    preload_model = os.environ.get("PRELOAD_MODEL", "lightgbm")
    preload_version = os.environ.get("PRELOAD_VERSION", "latest")
    if preload_model and preload_model.lower() != "none":
        try:
            logger.info("Warming up model '%s' (version: %s)...", preload_model, preload_version)
            load_and_cache_model(preload_model, preload_version)
        except Exception as e:
            logger.warning("Startup model warmup skipped or failed: %s", e)
    yield
    _MODEL_CACHE.clear()


app = FastAPI(
    title="TFG Real Estate Valuation - Inference Service",
    description="Ultra-fast, minimal inference REST API optimized for Azure Container Apps (scale-to-zero).",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Readiness and liveness probe for Azure Container Apps."""
    return HealthResponse(
        status="healthy",
        service="tfg-models-inference",
        version=__version__,
    )


@app.get("/models", tags=["Models"])
def list_models() -> Dict[str, Any]:
    """Lists registered models and currently cached models."""
    return {
        "registered_models": list(MODEL_REGISTRY.keys()),
        "cached_models": [f"{m}:{v}" for m, v in _MODEL_CACHE.keys()],
    }


@app.get("/models/{model_name}/versions", tags=["Models"])
def list_versions(model_name: str) -> Dict[str, Any]:
    """Lists available versions for a model in storage."""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}",
        )
    handler = get_model_handler(model_name)
    try:
        versions = handler.list_versions()
        return {"model": model_name, "versions": versions}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing versions for '{model_name}': {str(e)}",
        )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Performs real-time property valuation."""
    model, features, encoding = load_and_cache_model(request.model, request.version)

    try:
        X_input = prepare_inference_features(
            surface=request.surface,
            rooms=request.rooms,
            bathrooms=request.bathrooms,
            postal_code=request.postal_code,
            elevator=request.elevator,
            terrace=request.terrace,
            garage=request.garage,
            features=features,
            encoding=encoding,
        )
        raw_prediction = float(model.predict(X_input)[0])
    except Exception as e:
        logger.error("Prediction execution failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing prediction: {str(e)}",
        )

    return PredictResponse(
        model=request.model,
        version=request.version,
        estimated_price=round(raw_prediction, 2),
        currency="EUR",
        inputs=request.model_dump(),
    )


def start() -> None:
    """CLI script entrypoint to start inference server."""
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("tfg_models.api.inference:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start()
