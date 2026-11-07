"""Main pipeline orchestrator. Run end-to-end ML pipeline."""

import argparse
import logging
import json
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (DATA_RAW_DIR, RESULTS_DIR, FIGURES_DIR, EDA_FIGURES_DIR,
                         MODELS_DIR, SPLITS_DIR, TARGET_COL, GRADE_LABELS_5CLASS,
                         GRADE_LABELS_3CLASS, FIVE_TO_THREE_MAP)
from src.utils import setup_logging, stratified_split, create_label_sets, NumpyEncoder
from src.data_loader import load_raw_data, get_column_groups
from src.preprocessing import build_preprocessor, fit_and_save_preprocessor
from src.feature_engineering import build_feature_sets, compute_spearman_correlation
from src.train import (run_cv_experiment, select_best_model, train_final_model,
                        save_cv_results, run_binary_cv, run_vandijk_replication,
                        select_best_binary_model, _configure_model)
from src.evaluate import evaluate_on_test, evaluate_binary_on_test
from src.explain import compute_shap_values, compute_partial_dependence
from src.models import TwoStageORN

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def run_eda(df, column_groups):
    """Phase 2: EDA — generate summary stats and figures."""
    logger.info("=" * 60)
    logger.info("PHASE 2: EDA")
    logger.info("=" * 60)

    target = df[TARGET_COL]

    # Class distributions
    dist_5class = target.value_counts().sort_index().to_dict()
    dist_3class = pd.Series([FIVE_TO_THREE_MAP[g] for g in target]).value_counts().sort_index().to_dict()

    # Per-feature stats stratified by grade
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL and c != 'ORN_binary']

    stratified_stats = {}
    for col in numeric_cols[:30]:  # Limit to avoid huge output
        stats_per_grade = {}
        for grade in sorted(target.unique()):
            subset = df[df[TARGET_COL] == grade][col].dropna()
            stats_per_grade[str(int(grade))] = {
                "mean": float(subset.mean()) if len(subset) > 0 else None,
                "std": float(subset.std()) if len(subset) > 0 else None,
                "median": float(subset.median()) if len(subset) > 0 else None,
                "min": float(subset.min()) if len(subset) > 0 else None,
                "max": float(subset.max()) if len(subset) > 0 else None,
                "n": int(len(subset)),
            }
        stratified_stats[col] = stats_per_grade

    # Missing data summary
    missing = df.isnull().sum()
    missing_summary = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}

    # Correlation matrix of DVH features
    dvh_cols = column_groups.get("dvh", [])
    corr_matrix = None
    if dvh_cols:
        corr_df = compute_spearman_correlation(df, dvh_cols)
        corr_matrix = corr_df.to_dict()

    # Save EDA summary
    eda_summary = {
        "n_patients": int(len(df)),
        "n_features": int(len(df.columns)),
        "class_distribution_5class": {str(k): int(v) for k, v in dist_5class.items()},
        "class_distribution_3class": {str(k): int(v) for k, v in dist_3class.items()},
        "orn_incidence_rate": float(1 - dist_5class.get(0, 0) / len(df)),
        "missing_data": missing_summary,
        "stratified_stats": stratified_stats,
        "column_groups": {k: v for k, v in column_groups.items()},
    }

    with open(RESULTS_DIR / "eda_summary.json", 'w') as f:
        json.dump(eda_summary, f, indent=2, cls=NumpyEncoder)
    logger.info("EDA summary saved.")

    # --- EDA Figures ---
    _plot_class_distribution(df, target)
    _plot_dvh_correlation(dvh_cols, df)
    _plot_feature_distributions(df, column_groups, target)

    return eda_summary


def _plot_class_distribution(df, target):
    """Plot ORN grade distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 5-class
    counts = target.value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values, color='#3B5BDB', alpha=0.8)
    axes[0].set_xlabel('ORN Grade')
    axes[0].set_ylabel('Count')
    axes[0].set_title('5-class Distribution')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontsize=10)

    # 3-class
    three_class = pd.Series([FIVE_TO_THREE_MAP[g] for g in target])
    counts_3 = three_class.value_counts().sort_index()
    labels_3 = ['None (0)', 'Mild (1-2)', 'Severe (3-4)']
    axes[1].bar(labels_3, counts_3.values, color='#2F5D50', alpha=0.8)
    axes[1].set_xlabel('ORN Severity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('3-class Distribution')
    for i, v in enumerate(counts_3.values):
        axes[1].text(i, v + 5, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_dvh_correlation(dvh_cols, df):
    """Plot DVH feature correlation heatmap."""
    if not dvh_cols or len(dvh_cols) < 2:
        return

    corr = df[dvh_cols].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr, cmap='RdBu_r', center=0, ax=ax,
                xticklabels=True, yticklabels=True, fmt='.1f',
                annot=len(dvh_cols) <= 20)
    ax.set_title('DVH Feature Correlation (Spearman)')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "dvh_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()


def _plot_feature_distributions(df, column_groups, target):
    """Plot distributions of key features stratified by grade."""
    key_features = []
    for group in ['demographic', 'anatomical']:
        for c in column_groups.get(group, []):
            if df[c].dtype in ['float64', 'int64', 'float32']:
                key_features.append(c)

    # Add a few DVH features
    dvh_cols = column_groups.get('dvh', [])
    for c in dvh_cols[:4]:
        key_features.append(c)

    if not key_features:
        return

    n_feats = min(len(key_features), 8)
    ncols = 4
    nrows = (n_feats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).flatten()

    grade_colors = ['#94A3B8', '#FCD34D', '#FB923C', '#EF4444', '#991B1B']

    for i, feat in enumerate(key_features[:n_feats]):
        ax = axes[i]
        for grade in sorted(target.unique()):
            subset = df[df[TARGET_COL] == grade][feat].dropna()
            if len(subset) > 0:
                ax.hist(subset, bins=20, alpha=0.5, label=f'Grade {int(grade)}',
                        color=grade_colors[int(grade)])
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)

    # Hide empty axes
    for i in range(n_feats, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Feature Distributions by ORN Grade', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_FIGURES_DIR / "feature_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()


def main(data_path: str):
    """Run the full pipeline."""
    setup_logging()

    # ========================================
    # PHASE 2: Data Loading & EDA
    # ========================================
    df = load_raw_data(data_path)

    # Copy raw data
    dest = DATA_RAW_DIR / Path(data_path).name
    if not dest.exists():
        shutil.copy2(data_path, dest)

    column_groups = get_column_groups(df)
    eda_summary = run_eda(df, column_groups)

    # ========================================
    # PHASE 3 & 4: Preprocessing and Splitting
    # ========================================
    logger.info("=" * 60)
    logger.info("PHASE 3-4: Preprocessing & Splitting")
    logger.info("=" * 60)

    # Drop target columns from features
    target_cols = column_groups.get("target", [])
    other_drop = column_groups.get("other", [])
    feature_cols_all = [c for c in df.columns if c not in target_cols and c not in other_drop]

    # Split first (before fitting preprocessor)
    train_df, val_df, test_df = stratified_split(df)

    # Create label sets
    train_labels = create_label_sets(train_df)
    val_labels = create_label_sets(val_df)
    test_labels = create_label_sets(test_df)

    # Build and fit preprocessor on train only
    preprocessor, feature_columns_used = build_preprocessor(df, column_groups)

    # Fit on training data
    preprocessor, feature_names = fit_and_save_preprocessor(
        preprocessor, train_df[feature_columns_used])

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(
            preprocessor.transform(train_df[feature_columns_used][:1]).shape[1])]

    feature_names = list(feature_names)

    # Transform all splits
    X_train = preprocessor.transform(train_df[feature_columns_used])
    X_val = preprocessor.transform(val_df[feature_columns_used])
    X_test = preprocessor.transform(test_df[feature_columns_used])

    logger.info(f"Transformed shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Save feature names
    with open(RESULTS_DIR / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)

    # Save test set data for the API
    test_data = test_df.copy()
    test_data['predicted_index'] = list(range(len(test_df)))
    test_data.to_csv(RESULTS_DIR / "test_set_data.csv", index=False)

    # ========================================
    # PHASE 5: Feature Engineering
    # ========================================
    logger.info("=" * 60)
    logger.info("PHASE 5: Feature Engineering")
    logger.info("=" * 60)

    feature_sets = build_feature_sets(df, column_groups, df[TARGET_COL])

    # ========================================
    # PHASE 7: Training Loop
    # ========================================
    logger.info("=" * 60)
    logger.info("PHASE 7: Training Loop (CV)")
    logger.info("=" * 60)

    all_cv_results = []

    # ---- ADD 1: BINARY BASELINE (primary) ----
    logger.info("=" * 60)
    logger.info("ADD 1: BINARY BASELINE — pipeline sanity check")
    logger.info("=" * 60)

    # (a) Van Dijk minimal model: LR with D30 + Dental_extraction only
    d30_idx = None
    dental_idx = None
    for i, fn in enumerate(feature_names):
        if 'D30' in fn and 'continuous' in fn:
            d30_idx = i
        if 'Dental_extraction' in fn:
            dental_idx = i
    if d30_idx is not None and dental_idx is not None:
        vandijk_result = run_vandijk_replication(
            X_train, train_labels["y_binary"], feature_names, d30_idx, dental_idx)
        all_cv_results.append(vandijk_result)
    else:
        logger.warning(f"Could not find D30 (idx={d30_idx}) or Dental_extraction "
                      f"(idx={dental_idx}) for van Dijk replication. "
                      f"Feature names sample: {feature_names[:10]}")

    # (b) Full-feature binary CV
    logger.info("--- Binary formulation, full features ---")
    binary_cv_results = run_binary_cv(
        X_train, train_labels["y_binary"], feature_names=feature_names)
    all_cv_results.extend(binary_cv_results)

    best_binary = select_best_binary_model(binary_cv_results)

    # ---- 5-class and 3-class (kept as documented findings) ----
    logger.info("--- 5-class formulation, Set A ---")
    results_5c_a = run_cv_experiment(
        X_train, train_labels["y_5class"],
        feature_set_name="set_a", formulation="5class",
        feature_names=feature_names
    )
    all_cv_results.extend(results_5c_a)

    logger.info("--- 3-class formulation, Set A ---")
    results_3c_a = run_cv_experiment(
        X_train, train_labels["y_3class"],
        feature_set_name="set_a", formulation="3class",
        feature_names=feature_names
    )
    all_cv_results.extend(results_3c_a)

    # Save all CV results
    cv_df = save_cv_results(all_cv_results)

    best_5c = select_best_model([r for r in all_cv_results if r['formulation'] == '5class'])
    best_3c = select_best_model([r for r in all_cv_results if r['formulation'] == '3class'])

    logger.info(f"Best binary: {best_binary['model']} (AUC={best_binary['auc_mean']:.4f})")
    logger.info(f"Best 5-class: {best_5c['model']} (macro-F1={best_5c['macro_f1_mean']:.4f})")
    logger.info(f"Best 3-class: {best_3c['model']} (macro-F1={best_3c['macro_f1_mean']:.4f})")

    # ========================================
    # PHASE 7b: Train final models
    # ========================================
    logger.info("=" * 60)
    logger.info("PHASE 7b: Training Final Models")
    logger.info("=" * 60)

    import joblib
    X_trainval = np.vstack([X_train, X_val])

    # --- Binary model (primary) ---
    y_trainval_binary = np.concatenate([train_labels["y_binary"], val_labels["y_binary"]])
    binary_model = _configure_model(best_binary['model'], best_binary['imbalance_strategy'])

    if best_binary['imbalance_strategy'] == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            _, counts = np.unique(y_trainval_binary, return_counts=True)
            k = min(5, counts.min() - 1)
            if k >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_bin_resampled, y_bin_resampled = smote.fit_resample(X_trainval, y_trainval_binary)
            else:
                X_bin_resampled, y_bin_resampled = X_trainval, y_trainval_binary
        except Exception:
            X_bin_resampled, y_bin_resampled = X_trainval, y_trainval_binary
    else:
        X_bin_resampled, y_bin_resampled = X_trainval, y_trainval_binary

    binary_model.fit(X_bin_resampled, y_bin_resampled)
    joblib.dump(binary_model, MODELS_DIR / "best_model_binary.joblib")
    logger.info("Binary model saved.")

    # --- 5-class and 3-class (secondary) ---
    y_trainval_5c = np.concatenate([train_labels["y_5class"], val_labels["y_5class"]])
    y_trainval_3c = np.concatenate([train_labels["y_3class"], val_labels["y_3class"]])

    final_model_5c = train_final_model(X_trainval, y_trainval_5c, best_5c)

    model_3c = _configure_model(best_3c['model'], best_3c['imbalance_strategy'])
    model_3c.fit(X_trainval, y_trainval_3c)
    joblib.dump(model_3c, MODELS_DIR / "best_model_3class.joblib")

    # ========================================
    # PHASE 8: Test Set Evaluation (SINGLE TOUCH)
    # ========================================
    logger.info("=" * 60)
    logger.info("PHASE 8: Test Set Evaluation (SINGLE TOUCH)")
    logger.info("=" * 60)

    y_test_binary = test_labels["y_binary"]
    y_test_5c = test_labels["y_5class"]
    y_test_3c = test_labels["y_3class"]

    # Binary evaluation (primary — determines pass/fail)
    logger.info("--- Binary evaluation ---")
    results_binary = evaluate_binary_on_test(binary_model, X_test, y_test_binary)
    binary_auc = results_binary['metrics']['auc_roc']
    binary_auc_ci = results_binary['bootstrap_ci']['auc']

    # Pass/fail check
    VANDIJK_BENCHMARK = 0.74
    binary_passed = binary_auc >= VANDIJK_BENCHMARK
    validation_result = {
        "status": "PASSED" if binary_passed else "FAILED",
        "auc": binary_auc,
        "auc_ci": binary_auc_ci,
        "benchmark": VANDIJK_BENCHMARK,
        "best_model": best_binary['model'],
        "imbalance_strategy": best_binary['imbalance_strategy'],
    }
    with open(RESULTS_DIR / "binary_validation.json", 'w') as f:
        json.dump(validation_result, f, indent=2)

    if binary_passed:
        logger.info(f"BINARY VALIDATION PASSED: AUC={binary_auc:.4f} >= {VANDIJK_BENCHMARK}")
    else:
        logger.warning(f"BINARY VALIDATION FAILED: AUC={binary_auc:.4f} < {VANDIJK_BENCHMARK}")
        logger.warning("Pipeline may have a bug. Check preprocessing and feature sets.")
        logger.warning(f"  Best binary model: {best_binary['model']}/{best_binary['imbalance_strategy']}")
        logger.warning(f"  AUC CI: {binary_auc_ci}")
        logger.warning(f"  N features: {X_test.shape[1]}, N test: {len(y_test_binary)}")
        logger.warning("Proceeding anyway to generate comparison data, but results are suspect.")

    # 5-class and 3-class evaluation (secondary, for comparison table)
    logger.info("--- 5-class evaluation ---")
    results_5c = evaluate_on_test(final_model_5c, X_test, y_test_5c,
                                   GRADE_LABELS_5CLASS, formulation="5class")
    logger.info("--- 3-class evaluation ---")
    results_3c = evaluate_on_test(model_3c, X_test, y_test_3c,
                                   GRADE_LABELS_3CLASS, formulation="3class")

    # ========================================
    # ADD 2: TWO-STAGE HIERARCHICAL MODEL
    # ========================================
    logger.info("=" * 60)
    logger.info("ADD 2: TWO-STAGE HIERARCHICAL MODEL")
    logger.info("=" * 60)

    # Stage 2: train severity model on ORN-positive patients only
    # Filter train+val to ORN-positive
    y_5c_trainval = np.concatenate([train_labels["y_5class"], val_labels["y_5class"]])
    orn_mask_trainval = y_trainval_binary == 1
    X_orn_trainval = X_trainval[orn_mask_trainval]
    # Severity: Mild (Grade 1-2) = 0, Severe (Grade 3-4) = 1
    y_severity_trainval = np.array([
        0 if g in (1, 2) else 1 for g in y_5c_trainval[orn_mask_trainval]
    ])

    logger.info(f"Stage 2 training set: {len(X_orn_trainval)} ORN-positive patients")
    unique, counts = np.unique(y_severity_trainval, return_counts=True)
    logger.info(f"  Mild (0): {dict(zip(unique, counts)).get(0, 0)}, "
               f"Severe (1): {dict(zip(unique, counts)).get(1, 0)}")

    # Try LR, RF, XGBoost with class_weight='balanced' for stage 2
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score as sk_f1

    best_s2_score = -1
    best_s2_name = None
    best_s2_model = None

    s2_candidates = {
        "logistic_regression": _configure_model("logistic_regression", "class_weight"),
        "random_forest": _configure_model("random_forest", "class_weight"),
        "xgboost": _configure_model("xgboost", "class_weight"),
    }

    skf = StratifiedKFold(n_splits=min(5, min(counts)), shuffle=True, random_state=42)

    for s2_name, s2_model in s2_candidates.items():
        fold_f1s = []
        try:
            for tr_idx, va_idx in skf.split(X_orn_trainval, y_severity_trainval):
                s2_model_copy = _configure_model(s2_name, "class_weight")
                s2_model_copy.fit(X_orn_trainval[tr_idx], y_severity_trainval[tr_idx])
                y_p = s2_model_copy.predict(X_orn_trainval[va_idx])
                fold_f1s.append(sk_f1(y_severity_trainval[va_idx], y_p, average='macro', zero_division=0))
            mean_f1 = np.mean(fold_f1s)
            logger.info(f"  Stage 2 {s2_name}: macro-F1={mean_f1:.4f}")
            if mean_f1 > best_s2_score:
                best_s2_score = mean_f1
                best_s2_name = s2_name
        except Exception as e:
            logger.warning(f"  Stage 2 {s2_name} failed: {e}")

    logger.info(f"Best stage 2: {best_s2_name} (macro-F1={best_s2_score:.4f})")

    # Train final stage 2 model on all ORN-positive train+val
    stage2_model = _configure_model(best_s2_name, "class_weight")
    stage2_model.fit(X_orn_trainval, y_severity_trainval)
    joblib.dump(stage2_model, MODELS_DIR / "stage2_severity.joblib")

    # Build two-stage wrapper
    two_stage = TwoStageORN(stage1_model=binary_model, stage2_model=stage2_model)
    joblib.dump(two_stage, MODELS_DIR / "two_stage_model.joblib")
    logger.info("Two-stage model saved.")

    # Evaluate two-stage end-to-end on test set
    # Map test y to 3-class: 0=No ORN, 1=Mild (Grade 1-2), 2=Severe (Grade 3-4)
    y_test_twostage = np.array([
        0 if g == 0 else (1 if g in (1, 2) else 2) for g in y_test_5c
    ])
    TWO_STAGE_LABELS = {0: "No ORN", 1: "Mild ORN", 2: "Severe ORN"}

    results_twostage = evaluate_on_test(
        two_stage, X_test, y_test_twostage,
        class_labels=TWO_STAGE_LABELS, formulation="two_stage"
    )

    # Save two-stage specific results
    with open(RESULTS_DIR / "two_stage_metrics.json", 'w') as f:
        json.dump(results_twostage, f, indent=2)

    # ========================================
    # PHASE 9: Explainability (on binary model — the primary one)
    # ========================================
    logger.info("=" * 60)
    logger.info("PHASE 9: SHAP Explainability (binary model)")
    logger.info("=" * 60)

    shap_results = compute_shap_values(
        binary_model, X_test, feature_names,
        class_labels={0: "No ORN", 1: "ORN"}
    )

    pdp_data = compute_partial_dependence(
        binary_model, X_trainval, feature_names, top_n=10
    )

    # ========================================
    # Save default test_metrics.json as binary (primary)
    # ========================================
    with open(RESULTS_DIR / "test_metrics.json", 'w') as f:
        json.dump(results_binary, f, indent=2)

    # ========================================
    # Summary
    # ========================================
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"BINARY BASELINE: {'PASSED' if binary_passed else 'FAILED'}")
    logger.info(f"  AUC:           {binary_auc:.4f} [{binary_auc_ci[0]:.4f}, {binary_auc_ci[1]:.4f}]")
    logger.info(f"  Sensitivity:   {results_binary['metrics']['sensitivity']:.4f}")
    logger.info(f"  Specificity:   {results_binary['metrics']['specificity']:.4f}")
    logger.info(f"  F1(ORN):       {results_binary['metrics']['f1_orn']:.4f}")
    logger.info(f"  Benchmark:     van Dijk et al. 2021 AUC=0.77")
    logger.info("")
    logger.info(f"TWO-STAGE MODEL:")
    logger.info(f"  Macro-F1:      {results_twostage['metrics']['macro_f1']:.4f} "
               f"CI: {results_twostage['bootstrap_ci']['macro_f1_ci']}")
    for label, v in results_twostage['per_class'].items():
        logger.info(f"  {label:15s} F1={v['f1']:.3f}")
    logger.info("")
    logger.info(f"SINGLE-MODEL 3-CLASS (for comparison):")
    logger.info(f"  Macro-F1:      {results_3c['metrics']['macro_f1']:.4f} "
               f"CI: {results_3c['bootstrap_ci']['macro_f1_ci']}")
    logger.info("")
    ts_mf1 = results_twostage['metrics']['macro_f1']
    sm_mf1 = results_3c['metrics']['macro_f1']
    if ts_mf1 > sm_mf1:
        logger.info(f"Two-stage BEATS single-model 3-class by {ts_mf1 - sm_mf1:.4f} macro-F1")
    else:
        logger.info(f"Two-stage DOES NOT beat single-model 3-class (delta={ts_mf1 - sm_mf1:.4f})")
    logger.info("")
    logger.info(f"5-CLASS (documented negative finding):")
    logger.info(f"  Macro-F1:      {results_5c['metrics']['macro_f1']:.4f}")
    logger.info(f"All outputs saved to {RESULTS_DIR.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORN Grade Prediction Pipeline")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the CSV dataset")
    args = parser.parse_args()
    main(args.data)
