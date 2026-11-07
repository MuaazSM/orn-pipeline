"""Data loading and validation for the ORN pipeline."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from .config import TARGET_COL, BINARY_TARGET_COL

logger = logging.getLogger(__name__)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the raw CSV dataset.

    Args:
        path: Path to the CSV file.

    Returns:
        Validated DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    _validate_schema(df)
    _log_summary(df)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Check that expected columns and value ranges are present."""
    # Check target column exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    # Check ORN Grade values
    valid_grades = {0, 1, 2, 3, 4}
    actual_grades = set(df[TARGET_COL].dropna().unique())
    if not actual_grades.issubset(valid_grades):
        unexpected = actual_grades - valid_grades
        raise ValueError(f"Unexpected ORN Grade values: {unexpected}")

    logger.info("Schema validation passed.")


def _log_summary(df: pd.DataFrame) -> None:
    """Log basic dataset summary."""
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Target distribution
    dist = df[TARGET_COL].value_counts().sort_index()
    logger.info(f"Target distribution:\n{dist}")

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        logger.info(f"Columns with missing values:\n{missing_cols}")
    else:
        logger.info("No missing values found.")


def get_column_groups(df: pd.DataFrame) -> dict:
    """Identify column groups dynamically from the dataset.

    Returns a dict with keys: demographic, clinical, anatomical, dvh, target, other.
    """
    columns = list(df.columns)

    # Known columns (we'll match flexibly since CSV column names may vary)
    target_cols = [c for c in columns if c.lower() in
                   [TARGET_COL.lower(), BINARY_TARGET_COL.lower(),
                    "orn grade", "orn_grade", "orn_binary", "orngrade"]]

    # DVH columns: V5-V70, D2-D99, mean_dose, min_dose, max_dose, etc.
    dvh_cols = []
    for c in columns:
        clow = c.lower().strip()
        # Match V## or D## patterns
        if (clow.startswith('v') and clow[1:].replace('.', '').isdigit()) or \
           (clow.startswith('d') and clow[1:].replace('.', '').isdigit()):
            dvh_cols.append(c)
        elif any(kw in clow for kw in ['mean_dose', 'min_dose', 'max_dose',
                                        'mean dose', 'min dose', 'max dose',
                                        'meandose', 'mindose', 'maxdose',
                                        'mingy', 'maxgy']):
            dvh_cols.append(c)
        elif clow == 'mean' or clow == 'mean_dose':
            # 'mean' column is mean dose
            dvh_cols.append(c)

    # Clinical columns (check first to avoid 'age' substring matching T_stage)
    clinical_keywords = ['subsite', 't_stage', 'tstage', 'n_stage', 'nstage',
                         'chemo', 'postop', 'post_op', 'dental', 'pre_rt',
                         'pre-rt', 'postoperative']
    clinical_cols = [c for c in columns if any(kw in c.lower() for kw in clinical_keywords)
                     and c not in target_cols and c not in dvh_cols]

    # Demographic columns (after clinical, so T_stage/N_stage don't match 'age')
    demo_keywords = ['gender', 'sex', 'smoking', 'pack_year', 'pack year',
                     'packyear']
    demo_cols = [c for c in columns if any(kw in c.lower() for kw in demo_keywords)
                 and c not in target_cols and c not in dvh_cols and c not in clinical_cols]
    # Add Age column explicitly (avoid substring matching 'age' in 'stage')
    for c in columns:
        if c.lower().strip() == 'age' and c not in demo_cols and c not in clinical_cols:
            demo_cols.append(c)

    # Anatomical
    anat_keywords = ['mandible', 'volume']
    anat_cols = [c for c in columns if any(kw in c.lower() for kw in anat_keywords)
                 and c not in target_cols and c not in dvh_cols
                 and c not in demo_cols and c not in clinical_cols]

    # Everything else (including ID which should be dropped)
    identified = set(target_cols + dvh_cols + demo_cols + clinical_cols + anat_cols)
    other_cols = [c for c in columns if c not in identified]
    # ID column should always be in 'other' (will be dropped by preprocessor)


    groups = {
        "target": target_cols,
        "dvh": dvh_cols,
        "demographic": demo_cols,
        "clinical": clinical_cols,
        "anatomical": anat_cols,
        "other": other_cols,
    }

    for group_name, cols in groups.items():
        logger.info(f"Column group '{group_name}': {len(cols)} columns — {cols}")

    return groups
