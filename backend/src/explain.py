"""SHAP explainability: global/per-class importance, waterfall, PDP."""

import logging
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from .config import RESULTS_DIR, FIGURES_DIR, MODELS_DIR, GRADE_LABELS_5CLASS

logger = logging.getLogger(__name__)


def compute_shap_values(model, X_test: np.ndarray, feature_names: list,
                         class_labels: dict = None, n_sample_patients: int = 10):
    """Compute SHAP values and save all artifacts.

    Args:
        model: Trained model (tree-based preferred).
        X_test: Test features.
        feature_names: Feature names.
        class_labels: Grade label mapping.
        n_sample_patients: Number of sample patients for waterfall plots.
    """
    if class_labels is None:
        class_labels = GRADE_LABELS_5CLASS

    # Use TreeExplainer for tree models, KernelExplainer as fallback
    inner_model = getattr(model, '_model', model)
    try:
        explainer = shap.TreeExplainer(inner_model)
        shap_values = explainer.shap_values(X_test)
        logger.info("Using TreeExplainer")
    except Exception as e:
        logger.warning(f"TreeExplainer failed ({e}), using KernelExplainer")
        background = shap.sample(X_test, min(100, len(X_test)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test)

    # shap_values shape: list of (n_samples, n_features) per class, OR (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values)  # (n_classes, n_samples, n_features)
    else:
        # Transpose if needed
        if shap_values.ndim == 3:
            shap_array = shap_values.transpose(2, 0, 1)  # (n_classes, n_samples, n_features)
        else:
            shap_array = shap_values[np.newaxis, :, :]

    n_classes = shap_array.shape[0]
    classes = sorted(class_labels.keys())[:n_classes]

    # --- Global importance: mean |SHAP| across all classes ---
    global_importance = np.mean(np.abs(shap_array), axis=(0, 1))  # (n_features,)
    global_importance_df = pd.DataFrame({
        'feature': feature_names[:len(global_importance)],
        'mean_abs_shap': global_importance,
    }).sort_values('mean_abs_shap', ascending=False)

    # --- Per-class importance ---
    per_class_importance = {}
    for i, c in enumerate(classes):
        label = class_labels[c]
        class_imp = np.mean(np.abs(shap_array[i]), axis=0)
        per_class_importance[label] = {
            fn: float(v) for fn, v in
            zip(feature_names[:len(class_imp)], class_imp)
        }

    # --- Sample patients for waterfall plots ---
    sample_indices = np.random.RandomState(42).choice(
        len(X_test), size=min(n_sample_patients, len(X_test)), replace=False)

    sample_patients = []
    for idx in sample_indices:
        patient = {
            "index": int(idx),
            "features": {fn: float(X_test[idx, j])
                         for j, fn in enumerate(feature_names[:X_test.shape[1]])},
            "shap_values": {},
        }
        for i, c in enumerate(classes):
            label = class_labels[c]
            patient["shap_values"][label] = {
                fn: float(shap_array[i, idx, j])
                for j, fn in enumerate(feature_names[:shap_array.shape[2]])
            }
        sample_patients.append(patient)

    # --- Save everything ---
    # Global importance
    global_imp_data = {
        "features": global_importance_df['feature'].tolist(),
        "importance": global_importance_df['mean_abs_shap'].tolist(),
    }
    with open(RESULTS_DIR / "shap_global_importance.json", 'w') as f:
        json.dump(global_imp_data, f, indent=2)

    # Per-class importance
    with open(RESULTS_DIR / "shap_per_class_importance.json", 'w') as f:
        json.dump(per_class_importance, f, indent=2)

    # Sample patients
    with open(RESULTS_DIR / "shap_sample_patients.json", 'w') as f:
        json.dump(sample_patients, f, indent=2)

    # Raw SHAP values (numpy)
    np.save(RESULTS_DIR / "shap_values.npy", shap_array)

    # Save figures
    _save_shap_figures(global_importance_df, shap_array, X_test,
                        feature_names, class_labels, classes)

    logger.info("SHAP analysis complete and saved.")

    return {
        "global_importance": global_imp_data,
        "per_class_importance": per_class_importance,
        "sample_patients": sample_patients,
    }


def compute_partial_dependence(model, X_train: np.ndarray, feature_names: list,
                                top_n: int = 10):
    """Compute partial dependence data for top features.

    Uses manual PDP computation to save as JSON for frontend.
    """
    inner_model = getattr(model, '_model', model)

    # Get top features by importance
    if hasattr(inner_model, 'feature_importances_'):
        importances = inner_model.feature_importances_
    else:
        # Use SHAP global importance if available
        try:
            with open(RESULTS_DIR / "shap_global_importance.json") as f:
                data = json.load(f)
            importances = np.array(data['importance'])
        except Exception:
            importances = np.ones(X_train.shape[1])

    top_indices = np.argsort(importances)[::-1][:top_n]

    pdp_data = {}

    for feat_idx in top_indices:
        if feat_idx >= len(feature_names):
            continue
        feat_name = feature_names[feat_idx]
        values = np.linspace(
            np.percentile(X_train[:, feat_idx], 5),
            np.percentile(X_train[:, feat_idx], 95),
            50
        )

        predictions = []
        for v in values:
            X_mod = X_train.copy()
            X_mod[:, feat_idx] = v
            proba = model.predict_proba(X_mod)
            mean_proba = proba.mean(axis=0)
            predictions.append(mean_proba.tolist())

        pdp_data[feat_name] = {
            "values": values.tolist(),
            "predictions": predictions,
            "feature_index": int(feat_idx),
        }

    with open(RESULTS_DIR / "partial_dependence.json", 'w') as f:
        json.dump(pdp_data, f, indent=2)

    logger.info(f"Partial dependence computed for {len(pdp_data)} features")
    return pdp_data


def _save_shap_figures(global_imp_df, shap_array, X_test, feature_names,
                        class_labels, classes):
    """Save static SHAP figures as fallback."""
    # Global importance bar chart
    top_15 = global_imp_df.head(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_15)), top_15['mean_abs_shap'].values, color='#3B5BDB')
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title('Global Feature Importance (SHAP)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_global_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Per-class importance heatmap
    n_classes = len(classes)
    top_features = global_imp_df.head(20)['feature'].tolist()
    feat_indices = [i for i, fn in enumerate(feature_names) if fn in top_features]

    if feat_indices and n_classes > 1:
        heatmap_data = np.zeros((n_classes, len(feat_indices)))
        for ci in range(n_classes):
            for fi, feat_idx in enumerate(feat_indices):
                if feat_idx < shap_array.shape[2]:
                    heatmap_data[ci, fi] = np.mean(np.abs(shap_array[ci, :, feat_idx]))

        fig, ax = plt.subplots(figsize=(14, 6))
        sns_labels_x = [feature_names[i] for i in feat_indices]
        sns_labels_y = [class_labels.get(classes[i], f"Class {i}") for i in range(n_classes)]

        import seaborn as sns
        sns.heatmap(heatmap_data, xticklabels=sns_labels_x, yticklabels=sns_labels_y,
                    cmap='YlOrRd', ax=ax, annot=True, fmt='.3f')
        ax.set_title('Per-class Feature Importance (mean |SHAP|)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_per_class_importance.png", dpi=150,
                    bbox_inches='tight')
        plt.close()

    logger.info("SHAP figures saved.")
