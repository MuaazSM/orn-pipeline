"""Data endpoints: sample patients, DVH curves."""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter()

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "results"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


@router.get("/data/sample-patients")
async def get_sample_patients(
    n: int = Query(20, ge=1, le=200),
    grade: int = Query(None, ge=0, le=4),
    subsite: str = Query(None),
):
    """Return a sample of test set patients."""
    path = RESULTS_DIR / "test_set_data.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Test set data not found.")

    df = pd.read_csv(path)

    # Apply filters
    if grade is not None:
        df = df[df["ORN Grade"] == grade]
    if subsite is not None:
        subsite_cols = [c for c in df.columns if 'subsite' in c.lower()]
        if subsite_cols:
            df = df[df[subsite_cols[0]] == subsite]

    # Sample
    if len(df) > n:
        df = df.sample(n=n, random_state=42)

    # Convert to records
    records = df.replace({np.nan: None}).to_dict(orient='records')

    return {
        "patients": records,
        "n": len(records),
        "total_available": len(df),
    }


@router.get("/data/dvh-curves")
async def get_dvh_curves(
    grade: int = Query(None, ge=0, le=4),
    n: int = Query(10, ge=1, le=50),
):
    """Return DVH curve data for plotting."""
    # Find the raw data file
    data_files = list(DATA_DIR.glob("*.csv"))
    if not data_files:
        raise HTTPException(status_code=404, detail="Raw data not found.")

    df = pd.read_csv(data_files[0])

    if grade is not None:
        df = df[df["ORN Grade"] == grade]

    if len(df) > n:
        df = df.sample(n=n, random_state=42)

    # Extract DVH columns (V5-V70)
    v_cols = sorted([c for c in df.columns if c.startswith('V') and
                     c[1:].replace('.', '').isdigit()],
                    key=lambda x: float(x[1:]))

    if not v_cols:
        return {"curves": [], "dose_points": []}

    dose_points = [float(c[1:]) for c in v_cols]

    curves = []
    for _, row in df.iterrows():
        curve = {
            "grade": int(row.get("ORN Grade", -1)),
            "values": [float(row[c]) if pd.notna(row[c]) else None for c in v_cols],
        }
        curves.append(curve)

    return {
        "curves": curves,
        "dose_points": dose_points,
        "n": len(curves),
    }


@router.get("/data/columns")
async def get_columns():
    """Return available columns and their types."""
    path = RESULTS_DIR / "eda_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="EDA data not found.")

    with open(path) as f:
        eda = json.load(f)

    return {
        "column_groups": eda.get("column_groups", {}),
        "n_features": eda.get("n_features", 0),
    }


@router.get("/data/feature-histogram")
async def get_feature_histogram(
    feature: str = Query(..., description="Feature name"),
    bins: int = Query(20, ge=5, le=100),
):
    """Return histogram data for a feature, stratified by grade."""
    data_files = list(DATA_DIR.glob("*.csv"))
    if not data_files:
        raise HTTPException(status_code=404, detail="Raw data not found.")

    df = pd.read_csv(data_files[0])

    if feature not in df.columns:
        raise HTTPException(status_code=404,
                           detail=f"Feature '{feature}' not found.")

    result = {"feature": feature, "grades": {}}

    col = df[feature].dropna()
    if col.dtype in ['float64', 'int64', 'float32']:
        bin_edges = np.linspace(col.min(), col.max(), bins + 1)
        for grade in sorted(df["ORN Grade"].unique()):
            subset = df[df["ORN Grade"] == grade][feature].dropna()
            counts, _ = np.histogram(subset, bins=bin_edges)
            result["grades"][str(int(grade))] = {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            }
    else:
        for grade in sorted(df["ORN Grade"].unique()):
            subset = df[df["ORN Grade"] == grade][feature].dropna()
            vc = subset.value_counts().to_dict()
            result["grades"][str(int(grade))] = {
                "values": {str(k): int(v) for k, v in vc.items()},
            }

    return result
