"""Prediction endpoints: multi-class, binary, and two-stage."""

import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _preprocess_input(request: PredictionRequest, state: dict) -> np.ndarray:
    """Shared input preprocessing for all predict endpoints."""
    preprocessor = state["preprocessor"]
    input_df = pd.DataFrame([request.features])

    expected_cols = []
    for name, trans, cols in preprocessor.transformers_:
        expected_cols.extend(cols)

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    return preprocessor.transform(input_df[expected_cols])


def _compute_shap(model, X, feature_names):
    """Compute SHAP values for a single prediction."""
    try:
        import shap
        inner_model = getattr(model, '_model', model)
        explainer = shap.TreeExplainer(inner_model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            result = {}
            for cls_idx, cls_sv in enumerate(sv):
                result[str(cls_idx)] = {
                    fn: float(cls_sv[0, j])
                    for j, fn in enumerate(feature_names[:cls_sv.shape[1]])
                }
            return result
        elif sv.ndim == 3:
            result = {}
            for cls_idx in range(sv.shape[2]):
                result[str(cls_idx)] = {
                    fn: float(sv[0, j, cls_idx])
                    for j, fn in enumerate(feature_names[:sv.shape[1]])
                }
            return result
        else:
            # Binary: sv is (1, n_features) for the positive class
            return {
                "1": {fn: float(sv[0, j]) for j, fn in enumerate(feature_names[:sv.shape[1]])}
            }
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return None


@router.post("/predict", response_model=PredictionResponse)
async def predict_grade(request: PredictionRequest):
    """Predict ORN grade using the primary (binary) model."""
    from api.main import get_app_state
    state = get_app_state()

    if not state.get("model_loaded"):
        raise HTTPException(status_code=503, detail="Model not loaded.")

    model = state["model"]
    feature_names = state["feature_names"]

    try:
        X = _preprocess_input(request, state)
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        probabilities = {str(i): float(p) for i, p in enumerate(proba)}

        shap_values_dict = _compute_shap(model, X, feature_names)

        confidence_note = ""
        if pred == 1 and proba[1] < 0.6:
            confidence_note = "Borderline ORN probability. Consider clinical context."

        return PredictionResponse(
            predicted_grade=pred,
            probabilities=probabilities,
            shap_values=shap_values_dict,
            confidence_note=confidence_note,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@router.post("/predict/two-stage")
async def predict_two_stage(request: PredictionRequest):
    """Two-stage prediction: binary gate -> severity stratification."""
    from api.main import get_app_state
    state = get_app_state()

    two_stage = state.get("two_stage_model")
    if two_stage is None:
        raise HTTPException(status_code=503, detail="Two-stage model not loaded.")

    feature_names = state["feature_names"]

    try:
        X = _preprocess_input(request, state)

        # Get staged results
        staged = two_stage.predict_staged(X)[0]

        # SHAP for stage 1 (binary)
        stage1_shap = _compute_shap(two_stage.stage1_model, X, feature_names)

        # SHAP for stage 2 if flagged
        stage2_shap = None
        if staged["stage1_flagged"]:
            stage2_shap = _compute_shap(two_stage.stage2_model, X, feature_names)

        return {
            "stage1_probability": staged["stage1_p_orn"],
            "stage1_flagged": staged["stage1_flagged"],
            "stage2_probability_mild": staged["stage2_p_mild"],
            "stage2_probability_severe": staged["stage2_p_severe"],
            "final_class": staged["final_class"],
            "final_probabilities": staged["final_probabilities"],
            "stage1_shap": stage1_shap,
            "stage2_shap": stage2_shap,
        }
    except Exception as e:
        logger.error(f"Two-stage prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
