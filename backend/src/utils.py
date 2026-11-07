"""Utility functions for the ORN pipeline."""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .config import (RANDOM_STATE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
                     SPLITS_DIR, TARGET_COL, FIVE_TO_THREE_MAP)

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def stratified_split(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Perform stratified 60/20/20 split. Save indices.

    Returns (train_df, val_df, test_df) and saves index CSVs.
    """
    y = df[target_col]

    # First split: 60% train, 40% temp
    train_df, temp_df = train_test_split(
        df, test_size=(1 - TRAIN_RATIO), stratify=y,
        random_state=RANDOM_STATE
    )

    # Second split: 50% of 40% = 20% val, 20% test
    val_ratio_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_of_temp),
        stratify=temp_df[target_col],
        random_state=RANDOM_STATE
    )

    # Save indices
    train_df.index.to_series().to_csv(SPLITS_DIR / "train_indices.csv", index=False)
    val_df.index.to_series().to_csv(SPLITS_DIR / "val_indices.csv", index=False)
    test_df.index.to_series().to_csv(SPLITS_DIR / "test_indices.csv", index=False)

    # Log stratification
    _log_stratification(df, train_df, val_df, test_df, target_col)

    return train_df, val_df, test_df


def _log_stratification(df, train_df, val_df, test_df, target_col):
    """Log and verify stratification."""
    logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, "
               f"test={len(test_df)}")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df[target_col].value_counts(normalize=True).sort_index()
        logger.info(f"  {name} distribution:\n{dist}")

    # Verify no overlap
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert train_idx.isdisjoint(val_idx), "Train/val overlap!"
    assert train_idx.isdisjoint(test_idx), "Train/test overlap!"
    assert val_idx.isdisjoint(test_idx), "Val/test overlap!"
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(df), "Missing samples!"

    logger.info("Split verification passed: no overlap, all samples accounted for.")


def create_label_sets(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Create 5-class, 3-class, and ordinal label arrays.

    Returns dict with keys: y_5class, y_3class, y_ordinal.
    """
    y_5class = df[target_col].values

    y_3class = np.array([FIVE_TO_THREE_MAP[g] for g in y_5class])

    # Binary: 0 = no ORN, 1 = any ORN (Grade 1-4)
    y_binary = (y_5class > 0).astype(int)

    # Ordinal is same as 5-class for mord
    y_ordinal = y_5class.copy()

    logger.info(f"Label sets created:")
    logger.info(f"  5-class: {np.unique(y_5class, return_counts=True)}")
    logger.info(f"  3-class: {np.unique(y_3class, return_counts=True)}")
    logger.info(f"  binary:  {np.unique(y_binary, return_counts=True)}")

    return {
        "y_5class": y_5class,
        "y_3class": y_3class,
        "y_binary": y_binary,
        "y_ordinal": y_ordinal,
    }


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
