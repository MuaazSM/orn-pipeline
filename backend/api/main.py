"""FastAPI application for ORN prediction serving."""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

from api.routes import predict, explain, results, data

logger = logging.getLogger(__name__)

# Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "outputs" / "models"
RESULTS_DIR = BACKEND_DIR / "outputs" / "results"

# Global state
app_state = {
    "model": None,
    "model_3class": None,
    "preprocessor": None,
    "feature_names": None,
    "model_loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor at startup."""
    import json

    model_path = MODELS_DIR / "best_model.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    feature_names_path = RESULTS_DIR / "feature_names.json"

    try:
        app_state["preprocessor"] = joblib.load(preprocessor_path)

        with open(feature_names_path) as f:
            app_state["feature_names"] = json.load(f)

        # Primary model: binary (or fallback to best_model.joblib)
        binary_path = MODELS_DIR / "best_model_binary.joblib"
        if binary_path.exists():
            app_state["model"] = joblib.load(binary_path)
        else:
            app_state["model"] = joblib.load(model_path)

        # Two-stage model
        two_stage_path = MODELS_DIR / "two_stage_model.joblib"
        if two_stage_path.exists():
            app_state["two_stage_model"] = joblib.load(two_stage_path)

        # Secondary models
        model_3c_path = MODELS_DIR / "best_model_3class.joblib"
        if model_3c_path.exists():
            app_state["model_3class"] = joblib.load(model_3c_path)

        app_state["model_loaded"] = True
        logger.info("Models loaded: binary, two-stage, 3-class.")
    except FileNotFoundError as e:
        logger.warning(f"Could not load model artifacts: {e}. "
                      "Run the pipeline first.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

    yield

    # Cleanup
    app_state.clear()


app = FastAPI(
    title="ORN Grade Prediction API",
    description="API for osteoradionecrosis grade prediction and explainability",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api")
app.include_router(explain.router, prefix="/api")
app.include_router(results.router, prefix="/api")
app.include_router(data.router, prefix="/api")


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": app_state.get("model_loaded", False),
        "version": "1.0.0",
    }


def get_app_state():
    """Accessor for app state (used by route modules)."""
    return app_state
