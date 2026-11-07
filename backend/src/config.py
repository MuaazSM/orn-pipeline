"""Central configuration for the ORN prediction pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"
SPLITS_DIR = OUTPUTS_DIR / "splits"
EDA_FIGURES_DIR = FIGURES_DIR / "eda"

# Ensure output directories exist
for d in [MODELS_DIR, FIGURES_DIR, RESULTS_DIR, SPLITS_DIR, EDA_FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# Data split ratios (60/20/20)
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Target variable
TARGET_COL = "ORN Grade"
BINARY_TARGET_COL = "ORN_binary"

# Grade mappings
GRADE_LABELS_5CLASS = {0: "Grade 0", 1: "Grade 1", 2: "Grade 2", 3: "Grade 3", 4: "Grade 4"}
GRADE_LABELS_3CLASS = {0: "None", 1: "Mild", 2: "Severe"}

# 5-class to 3-class mapping: Grade 0 -> None, Grade 1-2 -> Mild, Grade 3-4 -> Severe
FIVE_TO_THREE_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}

# Feature groups
DEMOGRAPHIC_COLS = ["age", "gender", "smoking_status", "pack_years"]
CLINICAL_COLS = ["Subsite", "T_stage", "N_stage", "chemotherapy",
                 "postoperative_RT", "pre_RT_dental_extraction"]
ANATOMICAL_COLS = ["mandible_volume"]

# DVH columns will be identified dynamically from the data
DVH_V_PREFIX = "V"
DVH_D_PREFIX = "D"

# Correlation threshold for feature reduction
CORRELATION_THRESHOLD = 0.85

# CV settings
N_CV_FOLDS = 5

# Bootstrap settings
N_BOOTSTRAP_ITERATIONS = 1000

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
FRONTEND_URL = "http://localhost:3000"
