"""FastAPI Training and Experimentation Service."""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from tfg_models import __version__
from tfg_models.api.schemas import (
    CompareResponse,
    HealthResponse,
    ModelComparisonItem,
    ModelTrainResult,
    TrainRequest,
    TrainResponse,
)
from tfg_models.core.model_handler import get_model_handler
from tfg_models.models import MODEL_REGISTRY, get_model_trainer

logger = logging.getLogger(__name__)

# In-memory storage for background job statuses
_TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}


def _execute_training(
    targets: List[str], version: Optional[str], job_id: Optional[str] = None
) -> List[ModelTrainResult]:
    if job_id:
        _TRAINING_JOBS[job_id]["status"] = "running"

    results: List[ModelTrainResult] = []
    try:
        for model_name in targets:
            logger.info("Executing pipeline for model '%s'...", model_name)
            trainer = get_model_trainer(model_name)
            _, _, report, version_id = trainer.run(version=version)

            results.append(
                ModelTrainResult(
                    model=model_name,
                    version=version_id,
                    metrics={
                        "mae": report.get("mae", 0.0),
                        "rmse": report.get("rmse", 0.0),
                        "r2_score": report.get("r2_score", 0.0),
                    },
                )
            )

        if job_id:
            _TRAINING_JOBS[job_id]["status"] = "completed"
            _TRAINING_JOBS[job_id]["results"] = [r.model_dump() for r in results]
    except Exception as e:
        logger.error("Training execution failed: %s", e)
        if job_id:
            _TRAINING_JOBS[job_id]["status"] = "failed"
            _TRAINING_JOBS[job_id]["error"] = str(e)
        raise

    return results


app = FastAPI(
    title="TFG Real Estate Valuation - Training Service",
    description="HTTP API for triggering training pipelines, tracking metrics, and comparing models.",
    version=__version__,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Readiness probe for training service."""
    return HealthResponse(
        status="healthy",
        service="tfg-models-training",
        version=__version__,
    )


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train_models(request: TrainRequest, background_tasks: BackgroundTasks) -> TrainResponse:
    """Triggers ML training pipeline for specified model or all models."""
    if request.model != "all" and request.model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: '{request.model}'. Choose from: {['all'] + list(MODEL_REGISTRY.keys())}",
        )

    targets = list(MODEL_REGISTRY.keys()) if request.model == "all" else [request.model]

    if request.background:
        job_id = str(uuid.uuid4())
        _TRAINING_JOBS[job_id] = {
            "status": "pending",
            "models": targets,
            "version": request.version,
        }
        background_tasks.add_task(_execute_training, targets, request.version, job_id)
        return TrainResponse(
            status="accepted",
            message=f"Training job started in background for models: {targets}",
            job_id=job_id,
        )

    try:
        results = _execute_training(targets, request.version)
        return TrainResponse(
            status="completed",
            message=f"Training pipeline finished successfully for models: {targets}",
            results=results,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training execution error: {str(e)}",
        )


@app.get("/train/jobs/{job_id}", tags=["Training"])
def get_job_status(job_id: str) -> Dict[str, Any]:
    """Retrieves status and results of a background training job."""
    if job_id not in _TRAINING_JOBS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return _TRAINING_JOBS[job_id]


@app.get("/compare", response_model=CompareResponse, tags=["Evaluation"])
def compare_models() -> CompareResponse:
    """Compares benchmark metrics for latest versions of registered models."""
    items: List[ModelComparisonItem] = []
    for model_name in MODEL_REGISTRY:
        try:
            handler = get_model_handler(model_name)
            report = handler.get_report(version="latest")
            items.append(
                ModelComparisonItem(
                    model=model_name,
                    version=str(report.get("version", "latest")),
                    mae=report.get("mae"),
                    rmse=report.get("rmse"),
                    r2_score=report.get("r2_score"),
                    status="available",
                )
            )
        except Exception:
            items.append(
                ModelComparisonItem(
                    model=model_name,
                    version="none",
                    status="not_found",
                )
            )
    return CompareResponse(models=items)


@app.get("/models/{model_name}/report", tags=["Evaluation"])
def get_model_report(model_name: str, version: str = "latest") -> Dict[str, Any]:
    """Retrieves detailed evaluation metrics report for a model version."""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}",
        )
    handler = get_model_handler(model_name)
    try:
        return handler.get_report(version=version)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report not found for model '{model_name}' (version '{version}'): {str(e)}",
        )


def start() -> None:
    """CLI script entrypoint to start training server."""
    port = int(os.environ.get("PORT", "8001"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("tfg_models.api.training:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    start()
