"""Pydantic request and response schemas for TFG valuation HTTP APIs."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PropertyPayload(BaseModel):
    """Property characteristics for valuation inference."""

    surface: float = Field(..., gt=0, description="Surface area in square meters", examples=[95.0])
    rooms: int = Field(..., ge=0, description="Number of rooms / bedrooms", examples=[3])
    bathrooms: int = Field(..., ge=0, description="Number of bathrooms", examples=[2])
    postal_code: str = Field(..., min_length=1, description="Postal code (e.g. '36211')", examples=["36211"])
    elevator: bool = Field(default=False, description="Whether the building has an elevator")
    terrace: bool = Field(default=False, description="Whether the property includes a terrace")
    garage: bool = Field(default=False, description="Whether the property includes a garage")


class PredictRequest(PropertyPayload):
    """Request schema for property valuation inference."""

    model: str = Field(
        default="lightgbm",
        description="Target model architecture to use for estimation ('lightgbm', 'linear_regression', or 'neural_network')",
    )
    version: str = Field(
        default="latest",
        description="Persisted model version ID (or 'latest')",
    )


class PredictResponse(BaseModel):
    """Response schema for property valuation inference."""

    model: str
    version: str
    estimated_price: float = Field(..., description="Estimated market valuation in Euros (€)")
    currency: str = Field(default="EUR", description="Currency of estimated valuation")
    inputs: Dict[str, Any] = Field(..., description="Summary of input features used for evaluation")


class HealthResponse(BaseModel):
    """Response schema for healthcheck endpoints."""

    status: str
    service: str
    version: str


class TrainRequest(BaseModel):
    """Request schema to trigger model training pipelines."""

    model: str = Field(
        default="all",
        description="Model to train ('all', 'lightgbm', 'linear_regression', 'neural_network')",
    )
    version: Optional[str] = Field(
        default=None,
        description="Optional custom version ID (defaults to UTC timestamp YYYYMMDD_HHMMSS)",
    )
    background: bool = Field(
        default=False,
        description="Whether to run the training task in the background (asynchronous job)",
    )


class ModelTrainResult(BaseModel):
    """Evaluation summary result for a trained model."""

    model: str
    version: str
    metrics: Dict[str, Any]


class TrainResponse(BaseModel):
    """Response schema for model training execution."""

    status: str
    message: str
    results: Optional[List[ModelTrainResult]] = None
    job_id: Optional[str] = None


class ModelComparisonItem(BaseModel):
    """Model comparison entry for benchmark reporting."""

    model: str
    version: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    status: str = "available"


class CompareResponse(BaseModel):
    """Response schema comparing available registered models."""

    models: List[ModelComparisonItem]
