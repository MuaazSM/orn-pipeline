"""Training loop: cross-validation, model selection, and final evaluation."""

import logging
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC, SMOTE
import joblib

from .config import (RANDOM_STATE, N_CV_FOLDS, MODELS_DIR, RESULTS_DIR,
                     FIVE_TO_THREE_MAP)
from .models import get_all_models, get_binary_models

logger = logging.getLogger(__name__)


def get_imbalance_strategies(has_categorical: bool = False):
    """Return imbalance strategies to try.

    Returns dict of name -> callable that wraps a model.
    """
    return {
        "none": None,
        "class_weight": "class_weight",  # Handled by model parameter
        "smote": "smote",
    }


def run_cv_experiment(X_train: np.ndarray, y_train: np.ndarray,
                      feature_set_name: str, formulation: str,
                      feature_names: list = None,
                      categorical_indices: list = None) -> list:
    """Run stratified K-fold CV for all model × imbalance combos.

    Args:
        X_train: Training features (preprocessed).
        y_train: Training labels.
        feature_set_name: Name of feature set (set_a, set_b, set_c).
        formulation: '5class' or '3class'.
        feature_names: Feature names for logging.
        categorical_indices: Indices of categorical features for SMOTE-NC.

    Returns:
        List of result dicts.
    """
    models = get_all_models()
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    results = []

    for model_name, model in models.items():
        for imb_name in ["none", "class_weight", "smote"]:
            # Skip class_weight for models that don't support it
            if imb_name == "class_weight" and model_name in ["ordinal_logistic", "dummy_baseline"]:
                continue
            # Skip SMOTE for dummy
            if imb_name == "smote" and model_name == "dummy_baseline":
                continue

            logger.info(f"CV: {formulation} | {feature_set_name} | "
                       f"{model_name} | {imb_name}")

            fold_scores = {"macro_f1": [], "balanced_acc": [], "per_class_f1": []}

            try:
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                    # Apply imbalance strategy
                    model_instance = _configure_model(model_name, imb_name)

                    if imb_name == "smote":
                        # Apply SMOTE inside the fold
                        try:
                            # Check if smallest class has enough samples
                            _, counts = np.unique(y_fold_train, return_counts=True)
                            min_count = counts.min()
                            k_neighbors = min(5, min_count - 1)
                            if k_neighbors < 1:
                                logger.warning(f"Skipping SMOTE for fold {fold_idx}: "
                                             f"min class has {min_count} samples")
                                X_resampled, y_resampled = X_fold_train, y_fold_train
                            else:
                                smote = SMOTE(random_state=RANDOM_STATE,
                                            k_neighbors=k_neighbors)
                                X_resampled, y_resampled = smote.fit_resample(
                                    X_fold_train, y_fold_train)
                        except Exception as e:
                            logger.warning(f"SMOTE failed in fold {fold_idx}: {e}")
                            X_resampled, y_resampled = X_fold_train, y_fold_train
                    else:
                        X_resampled, y_resampled = X_fold_train, y_fold_train

                    model_instance.fit(X_resampled, y_resampled)
                    y_pred = model_instance.predict(X_fold_val)

                    fold_scores["macro_f1"].append(
                        f1_score(y_fold_val, y_pred, average='macro', zero_division=0))
                    fold_scores["balanced_acc"].append(
                        balanced_accuracy_score(y_fold_val, y_pred))

                    n_classes = len(np.unique(y_train))
                    per_class = f1_score(y_fold_val, y_pred, average=None,
                                         labels=list(range(n_classes)),
                                         zero_division=0)
                    fold_scores["per_class_f1"].append(per_class)

                # Aggregate
                result = {
                    "formulation": formulation,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "imbalance_strategy": imb_name,
                    "macro_f1_mean": float(np.mean(fold_scores["macro_f1"])),
                    "macro_f1_std": float(np.std(fold_scores["macro_f1"])),
                    "balanced_acc_mean": float(np.mean(fold_scores["balanced_acc"])),
                    "balanced_acc_std": float(np.std(fold_scores["balanced_acc"])),
                }

                per_class_arr = np.array(fold_scores["per_class_f1"])
                for c in range(per_class_arr.shape[1]):
                    result[f"class_{c}_f1_mean"] = float(np.mean(per_class_arr[:, c]))
                    result[f"class_{c}_f1_std"] = float(np.std(per_class_arr[:, c]))

                results.append(result)
                logger.info(f"  -> macro-F1: {result['macro_f1_mean']:.4f} "
                          f"± {result['macro_f1_std']:.4f}")

            except Exception as e:
                logger.error(f"Failed: {model_name} / {imb_name}: {e}")
                results.append({
                    "formulation": formulation,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "imbalance_strategy": imb_name,
                    "macro_f1_mean": 0.0,
                    "macro_f1_std": 0.0,
                    "balanced_acc_mean": 0.0,
                    "balanced_acc_std": 0.0,
                    "error": str(e),
                })

    return results


def _configure_model(model_name: str, imb_strategy: str):
    """Create a model instance with the appropriate imbalance config."""
    models = get_all_models()
    model = models[model_name]

    if imb_strategy == "class_weight":
        params = model.get_params()
        if 'class_weight' in params:
            model.set_params(class_weight='balanced')
        elif 'use_sample_weights' in params:
            model.set_params(use_sample_weights=True)
    elif imb_strategy == "none":
        params = model.get_params()
        if 'class_weight' in params:
            model.set_params(class_weight=None)
        if 'use_sample_weights' in params:
            model.set_params(use_sample_weights=False)

    return model


def select_best_model(cv_results: list) -> dict:
    """Select the best model configuration by macro-F1."""
    valid = [r for r in cv_results if 'error' not in r]
    if not valid:
        raise ValueError("No successful CV results found")

    best = max(valid, key=lambda r: r['macro_f1_mean'])
    logger.info(f"Best configuration: {best['model']} / {best['feature_set']} / "
               f"{best['imbalance_strategy']} "
               f"(macro-F1: {best['macro_f1_mean']:.4f})")
    return best


def train_final_model(X_trainval: np.ndarray, y_trainval: np.ndarray,
                       best_config: dict) -> object:
    """Refit best model on train+val and save."""
    model_name = best_config['model']
    imb_strategy = best_config['imbalance_strategy']

    model = _configure_model(model_name, imb_strategy)

    if imb_strategy == "smote":
        try:
            _, counts = np.unique(y_trainval, return_counts=True)
            k_neighbors = min(5, counts.min() - 1)
            if k_neighbors >= 1:
                smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X_trainval, y_trainval)
            else:
                X_resampled, y_resampled = X_trainval, y_trainval
        except Exception:
            X_resampled, y_resampled = X_trainval, y_trainval
    else:
        X_resampled, y_resampled = X_trainval, y_trainval

    model.fit(X_resampled, y_resampled)

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Final model saved to {model_path}")

    # Save best config
    config_path = RESULTS_DIR / "best_config.json"
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    return model


def run_binary_cv(X_train, y_train, feature_names=None):
    """Run binary-specific CV with AUC as primary metric."""
    from sklearn.metrics import roc_auc_score

    models = get_binary_models()
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)
    results = []

    for model_name, model in models.items():
        for imb_name in ["none", "class_weight", "smote"]:
            if imb_name == "class_weight" and model_name == "dummy_baseline":
                continue
            if imb_name == "smote" and model_name == "dummy_baseline":
                continue

            logger.info(f"Binary CV: {model_name} | {imb_name}")
            fold_auc = []
            fold_f1 = []
            fold_macro_f1 = []

            try:
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                    X_ft, X_fv = X_train[train_idx], X_train[val_idx]
                    y_ft, y_fv = y_train[train_idx], y_train[val_idx]

                    model_instance = _configure_model(model_name, imb_name)

                    if imb_name == "smote":
                        try:
                            _, counts = np.unique(y_ft, return_counts=True)
                            k = min(5, counts.min() - 1)
                            if k >= 1:
                                smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
                                X_ft, y_ft = smote.fit_resample(X_ft, y_ft)
                        except Exception:
                            pass

                    model_instance.fit(X_ft, y_ft)
                    y_pred = model_instance.predict(X_fv)
                    y_score = model_instance.predict_proba(X_fv)[:, 1]

                    fold_auc.append(roc_auc_score(y_fv, y_score))
                    fold_f1.append(f1_score(y_fv, y_pred, pos_label=1, zero_division=0))
                    fold_macro_f1.append(f1_score(y_fv, y_pred, average='macro', zero_division=0))

                result = {
                    "formulation": "binary",
                    "feature_set": "set_a",
                    "model": model_name,
                    "imbalance_strategy": imb_name,
                    "auc_mean": float(np.mean(fold_auc)),
                    "auc_std": float(np.std(fold_auc)),
                    "f1_orn_mean": float(np.mean(fold_f1)),
                    "f1_orn_std": float(np.std(fold_f1)),
                    "macro_f1_mean": float(np.mean(fold_macro_f1)),
                    "macro_f1_std": float(np.std(fold_macro_f1)),
                }
                results.append(result)
                logger.info(f"  -> AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f}, "
                          f"F1(ORN): {result['f1_orn_mean']:.4f}")

            except Exception as e:
                logger.error(f"Binary CV failed: {model_name}/{imb_name}: {e}")
                results.append({
                    "formulation": "binary", "feature_set": "set_a",
                    "model": model_name, "imbalance_strategy": imb_name,
                    "auc_mean": 0.0, "auc_std": 0.0,
                    "macro_f1_mean": 0.0, "macro_f1_std": 0.0,
                    "error": str(e),
                })

    return results


def run_vandijk_replication(X_train, y_train, feature_names, d30_idx, dental_idx):
    """Run van Dijk et al. 2021 minimal model: LR with D30 + Dental_extraction only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    logger.info("=== Van Dijk Replication: LR with D30 + Dental_extraction ===")

    X_minimal = X_train[:, [d30_idx, dental_idx]]

    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_auc = []

    for train_idx, val_idx in skf.split(X_minimal, y_train):
        lr = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE,
                                max_iter=1000)
        lr.fit(X_minimal[train_idx], y_train[train_idx])
        y_score = lr.predict_proba(X_minimal[val_idx])[:, 1]
        fold_auc.append(roc_auc_score(y_train[val_idx], y_score))

    result = {
        "formulation": "binary",
        "feature_set": "vandijk_minimal",
        "model": "logistic_regression_2feat",
        "imbalance_strategy": "class_weight",
        "features_used": ["D30", "Dental_extraction"],
        "auc_mean": float(np.mean(fold_auc)),
        "auc_std": float(np.std(fold_auc)),
        "macro_f1_mean": 0.0,  # placeholder, AUC is what matters here
        "macro_f1_std": 0.0,
    }
    logger.info(f"  Van Dijk replication AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
    logger.info(f"  Published benchmark: AUC ~0.77 (van Dijk et al. 2021)")
    return result


def select_best_binary_model(cv_results):
    """Select best binary model by AUC."""
    valid = [r for r in cv_results if 'error' not in r and r.get('auc_mean', 0) > 0]
    if not valid:
        raise ValueError("No successful binary CV results")
    best = max(valid, key=lambda r: r['auc_mean'])
    logger.info(f"Best binary: {best['model']} / {best['imbalance_strategy']} "
               f"(AUC: {best['auc_mean']:.4f})")
    return best


def save_cv_results(results: list):
    """Save all CV results to CSV and JSON."""
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "cv_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CV results saved to {csv_path}")

    json_path = RESULTS_DIR / "cv_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    return df
