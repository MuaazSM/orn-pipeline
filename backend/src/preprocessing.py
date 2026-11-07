"""Preprocessing pipeline for the ORN dataset."""

import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import joblib

from .config import MODELS_DIR

logger = logging.getLogger(__name__)

# Ordinal mappings
T_STAGE_ORDER = ["T1", "T2", "T3", "T4"]
N_STAGE_ORDER = ["N0", "N1", "N2", "N3"]


def build_preprocessor(df: pd.DataFrame, column_groups: dict) -> tuple:
    """Build a sklearn ColumnTransformer based on identified column groups.

    Args:
        df: The full dataframe (used to inspect categories).
        column_groups: Dict from get_column_groups().

    Returns:
        (preprocessor, feature_names_out, feature_columns_used)
    """
    transformers = []
    feature_columns_used = []

    # --- Continuous features: DVH + anatomical + age + pack_years ---
    continuous_cols = list(column_groups.get("dvh", []))
    continuous_cols += [c for c in column_groups.get("anatomical", []) if df[c].dtype in ['float64', 'int64', 'float32']]

    # Add numeric demographic columns
    for c in column_groups.get("demographic", []):
        if c.lower() in ['age', 'pack_years', 'pack_year', 'pack years'] or df[c].dtype in ['float64', 'float32']:
            if c not in continuous_cols:
                continuous_cols.append(c)

    if continuous_cols:
        continuous_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("continuous", continuous_pipeline, continuous_cols))
        feature_columns_used.extend(continuous_cols)

    # --- Ordinal features: T_stage, N_stage ---
    # Define ordinal ordering that captures the severity gradient
    T_STAGE_ORDERED = ["T0", "TX", "Tx", "T1", "T1a", "T1b", "T2", "T3", "T4", "T4a", "T4b"]
    N_STAGE_ORDERED = ["N0", "Nx", "N1", "N2", "N2a", "N2b", "N2c", "N3"]

    ordinal_cols = []
    ordinal_categories = []
    for c in column_groups.get("clinical", []):
        clow = c.lower().replace(' ', '_')
        if 't_stage' in clow or 'tstage' in clow:
            actual = df[c].dropna().unique().tolist()
            cats = [v for v in T_STAGE_ORDERED if v in actual]
            # Add any unseen categories at the end
            for v in actual:
                if v not in cats:
                    cats.append(v)
            ordinal_cols.append(c)
            ordinal_categories.append(cats)
        elif 'n_stage' in clow or 'nstage' in clow:
            actual = df[c].dropna().unique().tolist()
            cats = [v for v in N_STAGE_ORDERED if v in actual]
            for v in actual:
                if v not in cats:
                    cats.append(v)
            ordinal_cols.append(c)
            ordinal_categories.append(cats)

    if ordinal_cols:
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=ordinal_categories,
                                       handle_unknown="use_encoded_value",
                                       unknown_value=-1)),
        ])
        transformers.append(("ordinal", ordinal_pipeline, ordinal_cols))
        feature_columns_used.extend(ordinal_cols)

    # --- One-hot features: Subsite, chemotherapy, postop_RT, smoking_status, gender ---
    onehot_cols = []
    for c in column_groups.get("clinical", []):
        if c in ordinal_cols:
            continue
        clow = c.lower().replace(' ', '_')
        # postop_RT has string values ('postop_RT', 'Definitive'), not 0/1
        if any(kw in clow for kw in ['subsite', 'chemo', 'chemotherapy', 'postop']):
            onehot_cols.append(c)

    for c in column_groups.get("demographic", []):
        if c in continuous_cols:
            continue
        clow = c.lower().replace(' ', '_')
        if any(kw in clow for kw in ['gender', 'sex', 'smoking']):
            onehot_cols.append(c)

    if onehot_cols:
        onehot_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                                      drop=None)),
        ])
        transformers.append(("onehot", onehot_pipeline, onehot_cols))
        feature_columns_used.extend(onehot_cols)

    # --- Binary/remaining clinical features: Dental_extraction (already 0/1) ---
    binary_cols = []
    for c in column_groups.get("clinical", []):
        if c in ordinal_cols or c in onehot_cols:
            continue
        # Remaining clinical cols (e.g., Dental_extraction which is 0/1)
        binary_cols.append(c)

    if binary_cols:
        binary_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("binary", binary_pipeline, binary_cols))
        feature_columns_used.extend(binary_cols)

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info(f"Preprocessor built with {len(feature_columns_used)} input columns")
    logger.info(f"  Continuous: {len(continuous_cols)}, Ordinal: {len(ordinal_cols)}, "
                f"OneHot: {len(onehot_cols)}, Binary: {len(binary_cols)}")

    return preprocessor, feature_columns_used


def fit_and_save_preprocessor(preprocessor, X_train: pd.DataFrame, path=None):
    """Fit the preprocessor on training data and save it."""
    preprocessor.fit(X_train)
    save_path = path or MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, save_path)
    logger.info(f"Preprocessor saved to {save_path}")

    # Get feature names out
    try:
        feature_names = preprocessor.get_feature_names_out()
        logger.info(f"Output features: {len(feature_names)}")
    except Exception:
        feature_names = None
        logger.warning("Could not get feature names from preprocessor")

    return preprocessor, feature_names


def load_preprocessor(path=None):
    """Load a saved preprocessor."""
    load_path = path or MODELS_DIR / "preprocessor.joblib"
    return joblib.load(load_path)
