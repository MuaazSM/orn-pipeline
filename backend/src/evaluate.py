"""Evaluation: test set metrics, bootstrap CIs, curves."""

import logging
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    balanced_accuracy_score, cohen_kappa_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, brier_score_loss, recall_score, precision_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (RANDOM_STATE, N_BOOTSTRAP_ITERATIONS, RESULTS_DIR,
                     FIGURES_DIR, GRADE_LABELS_5CLASS)

logger = logging.getLogger(__name__)


def evaluate_on_test(model, X_test: np.ndarray, y_test: np.ndarray,
                      class_labels: dict = None, formulation: str = "5class"):
    """Full evaluation on the held-out test set.

    Returns a dict with all metrics + saves artifacts.
    """
    if class_labels is None:
        class_labels = GRADE_LABELS_5CLASS

    classes = sorted(class_labels.keys())
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # --- Core metrics ---
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')

    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=classes,
                             zero_division=0)
    report = classification_report(y_test, y_pred, labels=classes,
                                    target_names=[class_labels[c] for c in classes],
                                    output_dict=True, zero_division=0)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    # --- Bootstrap CIs ---
    logger.info(f"Running {N_BOOTSTRAP_ITERATIONS} bootstrap iterations...")
    bootstrap_metrics = _bootstrap_ci(model, X_test, y_test, classes)

    # --- ROC and PR curves ---
    curve_data = _compute_curves(y_test, y_proba, classes, class_labels)

    # --- Compile results ---
    results = {
        "formulation": formulation,
        "n_test": int(len(y_test)),
        "metrics": {
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "balanced_accuracy": float(bal_acc),
            "quadratic_weighted_kappa": float(qwk),
        },
        "bootstrap_ci": bootstrap_metrics,
        "per_class": {},
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "normalized": cm_normalized.tolist(),
            "labels": [class_labels[c] for c in classes],
        },
        "curves": curve_data,
        "classification_report": report,
    }

    for i, c in enumerate(classes):
        label = class_labels[c]
        results["per_class"][label] = {
            "f1": float(per_class_f1[i]),
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "support": int(report[label]["support"]),
            "f1_ci": bootstrap_metrics.get(f"class_{c}_f1_ci", [0, 0]),
        }

    # --- Save ---
    _save_results(results, formulation)
    _save_figures(cm, cm_normalized, class_labels, classes, formulation)

    logger.info(f"Test evaluation complete. Macro-F1: {macro_f1:.4f}, "
               f"Balanced Acc: {bal_acc:.4f}, QWK: {qwk:.4f}")

    return results


def _bootstrap_ci(model, X_test, y_test, classes, n_iter=None):
    """Compute bootstrap 95% CIs for key metrics."""
    if n_iter is None:
        n_iter = N_BOOTSTRAP_ITERATIONS

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_test)

    boot_macro_f1 = []
    boot_bal_acc = []
    boot_qwk = []
    boot_per_class = {c: [] for c in classes}

    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        X_b, y_b = X_test[idx], y_test[idx]
        y_pred_b = model.predict(X_b)

        boot_macro_f1.append(f1_score(y_b, y_pred_b, average='macro', zero_division=0))
        boot_bal_acc.append(balanced_accuracy_score(y_b, y_pred_b))
        boot_qwk.append(cohen_kappa_score(y_b, y_pred_b, weights='quadratic'))

        per_class = f1_score(y_b, y_pred_b, average=None, labels=classes, zero_division=0)
        for i, c in enumerate(classes):
            boot_per_class[c].append(per_class[i])

    ci = lambda arr: [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    result = {
        "macro_f1_ci": ci(boot_macro_f1),
        "balanced_accuracy_ci": ci(boot_bal_acc),
        "qwk_ci": ci(boot_qwk),
    }

    for c in classes:
        result[f"class_{c}_f1_ci"] = ci(boot_per_class[c])

    return result


def _compute_curves(y_test, y_proba, classes, class_labels):
    """Compute ROC and PR curve data for each class (one-vs-rest)."""
    y_bin = label_binarize(y_test, classes=classes)
    if y_bin.shape[1] == 1:
        # Binary case
        y_bin = np.hstack([1 - y_bin, y_bin])

    curve_data = {"roc": {}, "pr": {}}

    for i, c in enumerate(classes):
        label = class_labels[c]
        # ROC
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        curve_data["roc"][label] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "auc": float(roc_auc),
        }

        # PR
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        curve_data["pr"][label] = {
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
            "ap": float(ap),
        }

    return curve_data


def _save_results(results, formulation):
    """Save metrics to JSON."""
    path = RESULTS_DIR / f"test_metrics_{formulation}.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Test metrics saved to {path}")

    # Also save as the default test_metrics.json
    default_path = RESULTS_DIR / "test_metrics.json"
    with open(default_path, 'w') as f:
        json.dump(results, f, indent=2)


def _save_figures(cm, cm_normalized, class_labels, classes, formulation):
    """Save confusion matrix figures."""
    labels = [class_labels[c] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels,
                yticklabels=labels, ax=axes[0])
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_ylabel('True Grade')
    axes[0].set_xlabel('Predicted Grade')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Confusion Matrix (normalized)')
    axes[1].set_ylabel('True Grade')
    axes[1].set_xlabel('Predicted Grade')

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"confusion_matrix_{formulation}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix figure saved to {fig_path}")


def evaluate_binary_on_test(model, X_test: np.ndarray, y_test: np.ndarray):
    """Full evaluation for binary ORN vs no-ORN classification.

    This is the primary evaluation — establishes whether signal exists.
    """
    BINARY_LABELS = {0: "No ORN", 1: "ORN"}

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    # Probability of positive class (ORN)
    y_score = y_proba[:, 1]

    # --- Core metrics ---
    auc_roc = roc_auc_score(y_test, y_score)
    f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_neg = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=1, zero_division=0)  # TPR
    specificity = recall_score(y_test, y_pred, pos_label=0, zero_division=0)  # TNR
    ppv = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    npv = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    brier = brier_score_loss(y_test, y_score)
    ap = average_precision_score(y_test, y_score)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # --- ROC curve ---
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)

    # --- PR curve ---
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_score)

    # --- Optimal threshold (Youden's J) ---
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    optimal_threshold = float(roc_thresholds[best_j_idx])
    y_pred_optimal = (y_score >= optimal_threshold).astype(int)
    sensitivity_optimal = recall_score(y_test, y_pred_optimal, pos_label=1, zero_division=0)
    specificity_optimal = recall_score(y_test, y_pred_optimal, pos_label=0, zero_division=0)

    # --- Bootstrap CIs ---
    logger.info(f"Running {N_BOOTSTRAP_ITERATIONS} bootstrap iterations (binary)...")
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_test)
    boot = {"auc": [], "f1": [], "sensitivity": [], "specificity": [],
            "ppv": [], "npv": [], "brier": [], "bal_acc": [], "ap": []}

    for _ in range(N_BOOTSTRAP_ITERATIONS):
        idx = rng.choice(n, size=n, replace=True)
        yb, sb, pb = y_test[idx], y_score[idx], y_pred[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot["auc"].append(roc_auc_score(yb, sb))
        boot["f1"].append(f1_score(yb, pb, pos_label=1, zero_division=0))
        boot["sensitivity"].append(recall_score(yb, pb, pos_label=1, zero_division=0))
        boot["specificity"].append(recall_score(yb, pb, pos_label=0, zero_division=0))
        boot["ppv"].append(precision_score(yb, pb, pos_label=1, zero_division=0))
        boot["npv"].append(precision_score(yb, pb, pos_label=0, zero_division=0))
        boot["brier"].append(brier_score_loss(yb, sb))
        boot["bal_acc"].append(balanced_accuracy_score(yb, pb))
        boot["ap"].append(average_precision_score(yb, sb))

    ci = lambda arr: [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if arr else [0, 0]

    bootstrap_ci = {k: ci(v) for k, v in boot.items()}

    # --- Prevalence for context ---
    prevalence = float(y_test.sum() / len(y_test))

    # --- Compile results ---
    results = {
        "formulation": "binary",
        "n_test": int(len(y_test)),
        "prevalence": prevalence,
        "metrics": {
            "auc_roc": float(auc_roc),
            "average_precision": float(ap),
            "macro_f1": float(macro_f1),
            "f1_orn": float(f1_pos),
            "f1_no_orn": float(f1_neg),
            "balanced_accuracy": float(bal_acc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
            "brier_score": float(brier),
        },
        "optimal_threshold": {
            "threshold": optimal_threshold,
            "sensitivity": float(sensitivity_optimal),
            "specificity": float(specificity_optimal),
        },
        "bootstrap_ci": bootstrap_ci,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": ["No ORN", "ORN"],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "curves": {
            "roc": {
                "fpr": [float(x) for x in fpr],
                "tpr": [float(x) for x in tpr],
                "auc": float(auc_roc),
            },
            "pr": {
                "precision": [float(x) for x in pr_precision],
                "recall": [float(x) for x in pr_recall],
                "ap": float(ap),
            },
        },
        "per_class": {
            "No ORN": {
                "f1": float(f1_neg),
                "precision": float(npv),
                "recall": float(specificity),
                "support": int(tn + fp),
            },
            "ORN": {
                "f1": float(f1_pos),
                "precision": float(ppv),
                "recall": float(sensitivity),
                "support": int(fn + tp),
            },
        },
        # van Dijk et al. 2021 benchmark for comparison
        "published_benchmark": {
            "source": "van Dijk et al. 2021 NTCP model",
            "note": "Binary ORN prediction (any grade). AUC reported on internal validation.",
            "reported_auc": 0.74,
        },
    }

    # --- Save ---
    path = RESULTS_DIR / "test_metrics_binary.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    # Also save as default (binary is now primary)
    default_path = RESULTS_DIR / "test_metrics.json"
    with open(default_path, 'w') as f:
        json.dump(results, f, indent=2)

    # --- Figures ---
    labels = ["No ORN", "ORN"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    _save_figures(cm, cm_norm, {0: "No ORN", 1: "ORN"}, [0, 1], "binary")

    logger.info(f"Binary evaluation: AUC={auc_roc:.4f} [{bootstrap_ci['auc'][0]:.4f}, {bootstrap_ci['auc'][1]:.4f}]")
    logger.info(f"  Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}")
    logger.info(f"  F1(ORN)={f1_pos:.4f}, Macro-F1={macro_f1:.4f}")
    logger.info(f"  Brier={brier:.4f}, AP={ap:.4f}")
    logger.info(f"  Published benchmark AUC: 0.74 (van Dijk et al. 2021)")

    return results
