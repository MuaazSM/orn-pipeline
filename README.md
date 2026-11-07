# ORN Grade Prediction Pipeline

Predicts osteoradionecrosis (ORN) severity grades (0-4) in head and neck cancer patients using clinical and dosimetric features. Built on 1,259 patients from MD Anderson (2005-2015). Gradient boosting models with SHAP explainability, served via FastAPI with a Next.js research dashboard.

## Architecture

```
                  ┌─────────────┐         ┌──────────────┐
                  │   Next.js   │  REST   │   FastAPI    │
  Browser ───────>│  Frontend   │────────>│   Backend    │
                  │  :3000      │         │   :8000      │
                  └─────────────┘         └──────┬───────┘
                                                 │
                                          ┌──────┴───────┐
                                          │  ML Pipeline │
                                          │  (offline)   │
                                          └──────┬───────┘
                                                 │
                                          ┌──────┴───────┐
                                          │   Saved      │
                                          │   Artifacts  │
                                          └──────────────┘
```

## Directory Structure

```
orn_pipeline/
├── backend/
│   ├── data/raw/              # Raw CSV data
│   ├── src/                   # ML pipeline modules
│   │   ├── config.py          # Central configuration
│   │   ├── data_loader.py     # Data loading & validation
│   │   ├── preprocessing.py   # sklearn ColumnTransformer pipeline
│   │   ├── feature_engineering.py  # Correlation analysis, feature sets
│   │   ├── models.py          # Model definitions (LR, RF, XGB, LGBM, Ordinal)
│   │   ├── train.py           # CV training loop & model selection
│   │   ├── evaluate.py        # Test set evaluation with bootstrap CIs
│   │   ├── explain.py         # SHAP explainability & PDP
│   │   └── utils.py           # Splitting, label creation, utilities
│   ├── api/                   # FastAPI application
│   │   ├── main.py            # App entry point, CORS, model loading
│   │   ├── schemas.py         # Pydantic models
│   │   └── routes/            # Endpoint modules
│   ├── outputs/               # Pipeline artifacts
│   │   ├── models/            # Saved model & preprocessor
│   │   ├── figures/           # Static plots
│   │   ├── results/           # JSON metrics, SHAP data, CV results
│   │   └── splits/            # Train/val/test indices
│   ├── tests/
│   ├── requirements.txt
│   └── run_pipeline.py        # Pipeline CLI entry point
├── frontend/                  # Next.js application
├── Makefile
└── README.md
```

## Setup

### Backend

```bash
cd orn_pipeline/backend
pip install -r requirements.txt
```

### Frontend

```bash
cd orn_pipeline/frontend
npm install
```

## Running

### 1. Train the ML pipeline (offline, one-time)

```bash
cd orn_pipeline
python backend/run_pipeline.py --data /path/to/Final_final_used_data_Anonomyzed.csv
```

### 2. Start both services

```bash
cd orn_pipeline
make dev
```

Or separately:

```bash
# Terminal 1: Backend
cd orn_pipeline/backend
uvicorn api.main:app --reload --port 8000

# Terminal 2: Frontend
cd orn_pipeline/frontend
npm run dev
```

Open http://localhost:3000.

## Modeling decisions

- **Why gradient boosting over neural nets**: With 1,259 rows of tabular data, tree-based models consistently outperform deep learning. Reber et al. demonstrated this on this exact dataset.
- **Class imbalance**: Grade 1 has only 22 patients. We use class weighting and SMOTE (applied inside CV folds only) to handle this. All metrics include bootstrap CIs.
- **Primary metric**: Macro-F1 (not accuracy) to avoid bias toward the majority class.
- **Test set discipline**: Held-out test set touched exactly once, after all model selection.

## Citations

- van Dijk LV, et al. "Normal tissue complication probability (NTCP) prediction model for osteoradionecrosis of the mandible in patients with head and neck cancer after radiation therapy." *Radiotherapy and Oncology*. 2021. Dataset: figshare.

## Limitations

- Single-institution data (MD Anderson). No external validation cohort.
- Severe class imbalance: Grade 1 (n=22) and Grade 2 (n=36) predictions unreliable.
- Retrospective data; may not generalize to prospective clinical use.
- DVH parameters are summary statistics of complex 3D dose distributions.

## Disclaimer

**Research prototype. Not validated for clinical use. Do not use to inform patient care decisions.**

## Related work in this repository

- `impact_study/` — implementation study of the IMPACT jaw-segmentation paper (data exploration, code audit, plan). See `impact_study/README.md`.
- `docs/` — Semester 5 presentation decks (PDF, plus the IMPACT deck as PowerPoint).
