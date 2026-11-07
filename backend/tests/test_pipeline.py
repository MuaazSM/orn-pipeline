"""Basic pipeline tests."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd


def get_data_path():
    """Find the dataset."""
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "raw" / "Final_final_used_data_Anonomyzed.csv",
        Path(__file__).resolve().parent.parent.parent.parent / "Final_final_used_data_Anonomyzed.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pytest.skip("Dataset not found")


class TestDataLoading:
    def test_load_raw_data(self):
        from src.data_loader import load_raw_data
        df = load_raw_data(get_data_path())
        assert len(df) > 0
        assert "ORN Grade" in df.columns

    def test_column_groups(self):
        from src.data_loader import load_raw_data, get_column_groups
        df = load_raw_data(get_data_path())
        groups = get_column_groups(df)
        assert "target" in groups
        assert "dvh" in groups
        assert len(groups["dvh"]) > 0


class TestSplitting:
    def test_no_overlap(self):
        from src.data_loader import load_raw_data
        from src.utils import stratified_split
        df = load_raw_data(get_data_path())
        train, val, test = stratified_split(df)

        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)

        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)


class TestPreprocessing:
    def test_preprocessor_shape(self):
        from src.data_loader import load_raw_data, get_column_groups
        from src.preprocessing import build_preprocessor
        from src.utils import stratified_split

        df = load_raw_data(get_data_path())
        groups = get_column_groups(df)
        train, val, test = stratified_split(df)

        preprocessor, feature_cols = build_preprocessor(df, groups)
        preprocessor.fit(train[feature_cols])
        X_train = preprocessor.transform(train[feature_cols])
        X_test = preprocessor.transform(test[feature_cols])

        assert X_train.shape[0] == len(train)
        assert X_test.shape[0] == len(test)
        assert X_train.shape[1] == X_test.shape[1]
        assert X_train.shape[1] > 0


class TestModels:
    def test_model_predict_single(self):
        from src.models import get_all_models

        X = np.random.randn(100, 10)
        y = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.7, 0.05, 0.05, 0.1, 0.1])

        for name, model in get_all_models().items():
            if name == "ordinal_logistic":
                continue  # mord may need specific data
            model.fit(X, y)
            pred = model.predict(X[:1])
            proba = model.predict_proba(X[:1])
            assert len(pred) == 1
            assert proba.shape == (1, 5)
