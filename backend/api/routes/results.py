"""Results endpoints: metrics, confusion matrix, curves, CV results."""

import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter()

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "results"


@router.get("/results/summary")
async def get_summary():
    """Headline metrics from test evaluation."""
    path = RESULTS_DIR / "test_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Test metrics not found. Run pipeline first.")

    with open(path) as f:
        data = json.load(f)

    return {
        "formulation": data.get("formulation"),
        "n_test": data.get("n_test"),
        "metrics": data.get("metrics"),
        "bootstrap_ci": data.get("bootstrap_ci"),
        "per_class": data.get("per_class"),
    }


@router.get("/results/cv")
async def get_cv_results():
    """Full cross-validation comparison table."""
    path = RESULTS_DIR / "cv_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="CV results not found.")

    with open(path) as f:
        data = json.load(f)

    return {
        "results": data,
        "n_configurations": len(data),
    }


@router.get("/results/confusion-matrix")
async def get_confusion_matrix():
    """Confusion matrix data."""
    path = RESULTS_DIR / "test_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Test metrics not found.")

    with open(path) as f:
        data = json.load(f)

    cm = data.get("confusion_matrix", {})
    return {
        "matrix": cm.get("matrix", []),
        "normalized": cm.get("normalized", []),
        "labels": cm.get("labels", []),
    }


@router.get("/results/curves")
async def get_curves():
    """ROC and PR curve point data."""
    path = RESULTS_DIR / "test_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Test metrics not found.")

    with open(path) as f:
        data = json.load(f)

    return data.get("curves", {"roc": {}, "pr": {}})


@router.get("/results/eda")
async def get_eda_summary():
    """EDA summary data."""
    path = RESULTS_DIR / "eda_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="EDA summary not found.")

    with open(path) as f:
        return json.load(f)


@router.get("/results/binary-validation")
async def get_binary_validation():
    """Binary baseline pass/fail status."""
    path = RESULTS_DIR / "binary_validation.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Binary validation not found.")

    with open(path) as f:
        return json.load(f)


@router.get("/results/two-stage")
async def get_two_stage_results():
    """Two-stage hierarchical model results."""
    path = RESULTS_DIR / "two_stage_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Two-stage results not found.")

    with open(path) as f:
        return json.load(f)


@router.get("/results/model-comparison")
async def get_model_comparison():
    """Side-by-side comparison of binary, two-stage, and multi-class models."""
    comparison = {}

    for name, filename in [
        ("binary", "test_metrics_binary.json"),
        ("two_stage", "two_stage_metrics.json"),
        ("three_class", "test_metrics_3class.json"),
        ("five_class", "test_metrics_5class.json"),
    ]:
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            comparison[name] = {
                "metrics": data.get("metrics", {}),
                "bootstrap_ci": data.get("bootstrap_ci", {}),
                "per_class": data.get("per_class", {}),
                "formulation": data.get("formulation"),
            }

    # Add validation status
    val_path = RESULTS_DIR / "binary_validation.json"
    if val_path.exists():
        with open(val_path) as f:
            comparison["binary_validation"] = json.load(f)

    return comparison
