"""Feature engineering: correlation analysis and feature set generation."""

import logging
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from .config import CORRELATION_THRESHOLD, RESULTS_DIR

logger = logging.getLogger(__name__)


def compute_spearman_correlation(df: pd.DataFrame, dvh_cols: list) -> pd.DataFrame:
    """Compute Spearman correlation matrix for DVH features."""
    corr_matrix = df[dvh_cols].corr(method='spearman')
    return corr_matrix


def find_highly_correlated_pairs(corr_matrix: pd.DataFrame,
                                  threshold: float = CORRELATION_THRESHOLD) -> list:
    """Find pairs of features with |Spearman r| > threshold."""
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = abs(corr_matrix.iloc[i, j])
            if r > threshold:
                pairs.append((cols[i], cols[j], round(r, 4)))
    pairs.sort(key=lambda x: -x[2])
    logger.info(f"Found {len(pairs)} correlated pairs above threshold {threshold}")
    return pairs


def select_from_correlated_pair(df: pd.DataFrame, col_a: str, col_b: str,
                                 target: pd.Series) -> str:
    """Keep the feature more associated with the target (by Spearman |r|)."""
    r_a = abs(stats.spearmanr(df[col_a], target, nan_policy='omit').statistic)
    r_b = abs(stats.spearmanr(df[col_b], target, nan_policy='omit').statistic)
    return col_a if r_a >= r_b else col_b


def build_feature_sets(df: pd.DataFrame, column_groups: dict,
                       target: pd.Series) -> dict:
    """Build three feature sets: A (all), B (reduced), C (PCA on DVH + clinical).

    Returns a dict with feature set names -> list of column names (or 'pca' marker).
    """
    dvh_cols = column_groups.get("dvh", [])
    non_dvh_cols = (column_groups.get("demographic", []) +
                    column_groups.get("clinical", []) +
                    column_groups.get("anatomical", []))

    all_feature_cols = non_dvh_cols + dvh_cols

    # --- Set A: All features ---
    set_a = list(all_feature_cols)

    # --- Set B: Reduced (drop one of each correlated pair) ---
    if dvh_cols:
        corr_matrix = compute_spearman_correlation(df, dvh_cols)
        pairs = find_highly_correlated_pairs(corr_matrix)

        to_drop = set()
        for col_a, col_b, r in pairs:
            if col_a in to_drop or col_b in to_drop:
                continue
            keep = select_from_correlated_pair(df, col_a, col_b, target)
            drop = col_b if keep == col_a else col_a
            to_drop.add(drop)

        set_b = [c for c in all_feature_cols if c not in to_drop]
        logger.info(f"Set B: dropped {len(to_drop)} correlated features, "
                    f"kept {len(set_b)} features")
    else:
        set_b = list(all_feature_cols)

    # --- Set C: PCA on DVH + clinical raw ---
    set_c_info = {
        "dvh_cols_for_pca": dvh_cols,
        "non_dvh_cols": non_dvh_cols,
        "pca_variance_threshold": 0.95,
    }

    feature_sets = {
        "set_a": {"columns": set_a, "description": "All features"},
        "set_b": {"columns": set_b, "description": "Reduced (correlated DVH removed)"},
        "set_c": {"pca_config": set_c_info,
                   "description": "PCA on DVH (95% variance) + clinical raw"},
    }

    # Save manifest
    manifest_path = RESULTS_DIR / "feature_sets.json"
    # Make JSON-serializable
    manifest = {}
    for k, v in feature_sets.items():
        manifest[k] = {
            "description": v["description"],
            "n_features": len(v.get("columns", [])) or "dynamic (PCA)",
        }
        if "columns" in v:
            manifest[k]["columns"] = v["columns"]
        if "pca_config" in v:
            manifest[k]["pca_config"] = v["pca_config"]

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Feature set manifest saved to {manifest_path}")

    return feature_sets


def apply_pca_feature_set(X_train_dvh: np.ndarray, X_test_dvh: np.ndarray,
                           variance_threshold: float = 0.95) -> tuple:
    """Apply PCA to DVH features, keeping enough components for threshold variance.

    Returns (pca, X_train_pca, X_test_pca).
    """
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_train_pca = pca.fit_transform(X_train_dvh)
    X_test_pca = pca.transform(X_test_dvh)

    logger.info(f"PCA: {X_train_dvh.shape[1]} DVH features -> "
                f"{pca.n_components_} components "
                f"(explained variance: {pca.explained_variance_ratio_.sum():.4f})")

    return pca, X_train_pca, X_test_pca
