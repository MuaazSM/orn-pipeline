"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class PredictionRequest(BaseModel):
    """Patient feature dict for prediction."""
    features: dict = Field(..., description="Dictionary of feature_name -> value")


class PredictionResponse(BaseModel):
    predicted_grade: int
    probabilities: dict
    shap_values: Optional[dict] = None
    confidence_note: str = ""


class MetricsSummaryResponse(BaseModel):
    formulation: str
    n_test: int
    metrics: dict
    bootstrap_ci: dict
    per_class: dict


class ConfusionMatrixResponse(BaseModel):
    matrix: list
    normalized: list
    labels: list


class CurveDataResponse(BaseModel):
    roc: dict
    pr: dict


class CVResultsResponse(BaseModel):
    results: list
    n_configurations: int


class GlobalSHAPResponse(BaseModel):
    features: list
    importance: list


class PerClassSHAPResponse(BaseModel):
    per_class: dict


class PartialDependenceResponse(BaseModel):
    feature: str
    values: list
    predictions: list


class SamplePatientsResponse(BaseModel):
    patients: list
    n: int
