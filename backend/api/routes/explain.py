"""Explainability endpoints: SHAP and PDP."""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "results"


@router.get("/explain/global")
async def get_global_shap():
    """Global SHAP feature importance."""
    path = RESULTS_DIR / "shap_global_importance.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="SHAP data not found. Run pipeline first.")

    with open(path) as f:
        return json.load(f)


@router.get("/explain/per-class")
async def get_per_class_shap():
    """Per-class SHAP importances."""
    path = RESULTS_DIR / "shap_per_class_importance.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Per-class SHAP data not found.")

    with open(path) as f:
        return json.load(f)


@router.get("/explain/partial-dependence")
async def get_partial_dependence(feature: str = Query(..., description="Feature name")):
    """Partial dependence data for a specific feature."""
    path = RESULTS_DIR / "partial_dependence.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDP data not found.")

    with open(path) as f:
        data = json.load(f)

    if feature not in data:
        available = list(data.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Feature '{feature}' not found. Available: {available}")

    return {"feature": feature, **data[feature]}


@router.get("/explain/sample-patients")
async def get_sample_patients_shap():
    """SHAP values for sample patients (for waterfall plots)."""
    path = RESULTS_DIR / "shap_sample_patients.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample patient SHAP data not found.")

    with open(path) as f:
        return json.load(f)
