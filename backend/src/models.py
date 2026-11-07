"""Model definitions with sklearn-compatible APIs."""

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import RANDOM_STATE

logger = logging.getLogger(__name__)


class LogisticRegressionMulticlass(BaseEstimator, ClassifierMixin):
    """Multinomial logistic regression with L2 regularization."""

    def __init__(self, C=1.0, max_iter=1000, class_weight='balanced'):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self._model = None

    def fit(self, X, y, **kwargs):
        self._model = LogisticRegression(
            C=self.C, penalty='l2',
            solver='lbfgs', max_iter=self.max_iter,
            class_weight=self.class_weight, random_state=RANDOM_STATE
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def get_params(self, deep=True):
        return {"C": self.C, "max_iter": self.max_iter,
                "class_weight": self.class_weight}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class RandomForestModel(BaseEstimator, ClassifierMixin):
    """Random Forest with balanced class weights."""

    def __init__(self, n_estimators=300, max_depth=None,
                 class_weight='balanced', min_samples_leaf=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.min_samples_leaf = min_samples_leaf
        self._model = None

    def fit(self, X, y, **kwargs):
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            class_weight=self.class_weight, min_samples_leaf=self.min_samples_leaf,
            random_state=RANDOM_STATE, n_jobs=-1
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "class_weight": self.class_weight,
                "min_samples_leaf": self.min_samples_leaf}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class XGBoostModel(BaseEstimator, ClassifierMixin):
    """XGBoost with sample weight support for imbalance."""

    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1,
                 scale_pos_weight=None, use_sample_weights=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.use_sample_weights = use_sample_weights
        self._model = None

    def _compute_sample_weights(self, y):
        """Compute sample weights inversely proportional to class frequency."""
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        weights_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        return np.array([weights_map[yi] for yi in y])

    def fit(self, X, y, **kwargs):
        self._model = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, objective='multi:softprob',
            eval_metric='mlogloss', random_state=RANDOM_STATE,
            use_label_encoder=False, n_jobs=-1, verbosity=0
        )
        fit_params = {}
        if self.use_sample_weights:
            fit_params['sample_weight'] = self._compute_sample_weights(y)
        self._model.fit(X, y, **fit_params)
        self.classes_ = self._model.classes_
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "scale_pos_weight": self.scale_pos_weight,
                "use_sample_weights": self.use_sample_weights}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class LightGBMModel(BaseEstimator, ClassifierMixin):
    """LightGBM with balanced class weights."""

    def __init__(self, n_estimators=300, max_depth=-1, learning_rate=0.1,
                 class_weight='balanced', num_leaves=31):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.num_leaves = num_leaves
        self._model = None

    def fit(self, X, y, **kwargs):
        self._model = LGBMClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, class_weight=self.class_weight,
            num_leaves=self.num_leaves, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1
        )
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        self.feature_importances_ = self._model.feature_importances_
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "class_weight": self.class_weight,
                "num_leaves": self.num_leaves}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class OrdinalLogisticModel(BaseEstimator, ClassifierMixin):
    """Ordinal logistic regression using mord.LogisticAT."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._model = None

    def fit(self, X, y, **kwargs):
        from mord import LogisticAT
        self._model = LogisticAT(alpha=self.alpha)
        self._model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        """Approximate probabilities from ordinal model."""
        # mord doesn't have predict_proba natively; use decision function
        try:
            # Get cumulative probabilities from the threshold model
            from scipy.special import expit
            theta = self._model.theta_
            coef = self._model.coef_
            linear = X @ coef.T if hasattr(coef, 'T') else X @ coef.reshape(-1, 1)
            linear = linear.ravel()

            n_classes = len(self.classes_)
            cumprobs = np.zeros((len(X), n_classes - 1))
            for i, t in enumerate(theta):
                cumprobs[:, i] = expit(t - linear)

            proba = np.zeros((len(X), n_classes))
            proba[:, 0] = cumprobs[:, 0]
            for i in range(1, n_classes - 1):
                proba[:, i] = cumprobs[:, i] - cumprobs[:, i - 1]
            proba[:, -1] = 1.0 - cumprobs[:, -1]
            proba = np.clip(proba, 0, 1)
            # Normalize
            row_sums = proba.sum(axis=1, keepdims=True)
            proba = proba / np.where(row_sums > 0, row_sums, 1)
            return proba
        except Exception as e:
            logger.warning(f"Could not compute probabilities for ordinal model: {e}")
            # Fallback: one-hot encode predictions
            preds = self.predict(X)
            proba = np.zeros((len(X), len(self.classes_)))
            for i, p in enumerate(preds):
                idx = np.where(self.classes_ == p)[0]
                if len(idx) > 0:
                    proba[i, idx[0]] = 1.0
            return proba

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class StratifiedDummyClassifier(BaseEstimator, ClassifierMixin):
    """Dummy classifier as baseline."""

    def __init__(self, strategy='stratified'):
        self.strategy = strategy
        self._model = None

    def fit(self, X, y, **kwargs):
        self._model = DummyClassifier(strategy=self.strategy,
                                       random_state=RANDOM_STATE)
        self._model.fit(X, y)
        self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def get_params(self, deep=True):
        return {"strategy": self.strategy}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


class TwoStageORN(BaseEstimator, ClassifierMixin):
    """Two-stage hierarchical model: binary gate → severity stratification.

    Stage 1: ORN vs no-ORN (binary)
    Stage 2: Mild (Grade 1-2) vs Severe (Grade 3-4), only among ORN-positive

    Final output: one of {0: No ORN, 1: Mild ORN, 2: Severe ORN}
    """

    def __init__(self, stage1_model=None, stage2_model=None, threshold=0.5):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.threshold = threshold
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y_binary, X_orn=None, y_severity=None):
        """Fit both stages.

        Args:
            X: Full feature matrix for stage 1.
            y_binary: Binary labels (0=no ORN, 1=ORN) for stage 1.
            X_orn: Features for ORN-positive patients only (stage 2).
            y_severity: Severity labels (0=Mild, 1=Severe) for stage 2.
        """
        self.stage1_model.fit(X, y_binary)

        if X_orn is not None and y_severity is not None:
            self.stage2_model.fit(X_orn, y_severity)

        return self

    def predict(self, X):
        """Predict final 3-class output."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """Predict calibrated 3-class probabilities.

        P(No ORN) = P(stage1=0)
        P(Mild)   = P(stage1=1) * P(stage2=Mild)
        P(Severe) = P(stage1=1) * P(stage2=Severe)
        """
        s1_proba = self.stage1_model.predict_proba(X)
        p_no_orn = s1_proba[:, 0]
        p_orn = s1_proba[:, 1]

        s2_proba = self.stage2_model.predict_proba(X)
        p_mild_given_orn = s2_proba[:, 0]
        p_severe_given_orn = s2_proba[:, 1]

        proba = np.column_stack([
            p_no_orn,
            p_orn * p_mild_given_orn,
            p_orn * p_severe_given_orn,
        ])
        return proba

    def predict_staged(self, X):
        """Return detailed per-stage results for the API."""
        s1_proba = self.stage1_model.predict_proba(X)
        p_orn = s1_proba[:, 1]

        results = []
        for i in range(len(X)):
            row = {
                "stage1_p_orn": float(p_orn[i]),
                "stage1_flagged": bool(p_orn[i] >= self.threshold),
            }
            if p_orn[i] >= self.threshold:
                s2_proba = self.stage2_model.predict_proba(X[i:i+1])[0]
                row["stage2_p_mild"] = float(s2_proba[0])
                row["stage2_p_severe"] = float(s2_proba[1])
                row["final_class"] = "Severe ORN" if s2_proba[1] > s2_proba[0] else "Mild ORN"
            else:
                row["stage2_p_mild"] = None
                row["stage2_p_severe"] = None
                row["final_class"] = "No ORN"

            full_proba = self.predict_proba(X[i:i+1])[0]
            row["final_probabilities"] = {
                "No ORN": float(full_proba[0]),
                "Mild ORN": float(full_proba[1]),
                "Severe ORN": float(full_proba[2]),
            }
            results.append(row)
        return results

    def get_params(self, deep=True):
        return {"threshold": self.threshold}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def get_all_models() -> dict:
    """Return a dict of model_name -> model_instance for all models."""
    return {
        "logistic_regression": LogisticRegressionMulticlass(),
        "random_forest": RandomForestModel(),
        "xgboost": XGBoostModel(),
        "lightgbm": LightGBMModel(),
        "ordinal_logistic": OrdinalLogisticModel(),
        "dummy_baseline": StratifiedDummyClassifier(),
    }


def get_binary_models() -> dict:
    """Models for binary classification (skip ordinal and dummy)."""
    return {
        "logistic_regression": LogisticRegressionMulticlass(),
        "random_forest": RandomForestModel(),
        "xgboost": XGBoostModel(),
        "lightgbm": LightGBMModel(),
        "dummy_baseline": StratifiedDummyClassifier(),
    }
