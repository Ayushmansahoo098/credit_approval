"""
main.py — Single entry point to run the full Credit Approval ML pipeline.

Usage:
    python main.py [--quick] [--trials N]

Options:
    --quick     Skip Optuna HPO; use default params (faster run for testing)
    --trials N  Number of Optuna trials per model (default: 60)
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DATA_PATH    = ROOT / "data" / "raw" / "crx.data"
MODELS_DIR   = ROOT / "outputs" / "models"
REPORTS_DIR  = ROOT / "outputs" / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from src.utils import load_raw_data, get_optimal_threshold, TARGET_COL
from src.preprocessing import build_preprocessing_pipeline, get_feature_names
from src.training import run_all_base_models, run_optuna_hpo, train_stacking_ensemble
from src.evaluation import (
    compute_all_metrics,
    plot_roc_curves, plot_pr_curves, plot_calibration, plot_ks_chart,
    plot_confusion_matrix, compute_shap_values, plot_shap_summary,
    plot_shap_dependence, fairness_analysis, build_results_table,
)
from src.models import get_stacking_ensemble, LIGHTGBM_PARAMS, XGBOOST_PARAMS


def parse_args():
    p = argparse.ArgumentParser(description="Credit Approval ML Pipeline")
    p.add_argument("--quick",  action="store_true",
                   help="Skip HPO; use default params")
    p.add_argument("--trials", type=int, default=60,
                   help="Optuna trials per model (default: 60)")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  Credit Approval ML Pipeline")
    print("=" * 60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    df = load_raw_data(str(DATA_PATH))
    X_raw = df.drop(columns=[TARGET_COL])
    y     = df[TARGET_COL].values
    print(f"  Shape: {df.shape}  |  Approved: {y.sum()} ({y.mean():.1%})")

    # Hold-out split for final evaluation + fairness analysis
    X_tr_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.15, stratify=y, random_state=42
    )

    # ── 2. Preprocessing ──────────────────────────────────────────────────────
    print("\n[2/7] Building preprocessing pipeline...")
    preproc = build_preprocessing_pipeline()
    X_train = preproc.fit_transform(X_tr_raw, y_train)
    X_test  = preproc.transform(X_test_raw)
    feature_names = get_feature_names(preproc)
    print(f"  Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")
    joblib.dump(preproc, MODELS_DIR / "preprocessor.pkl")

    # ── 3. Base model CV ──────────────────────────────────────────────────────
    print("\n[3/7] Training base models (10-fold CV with SMOTEENN)...")
    cv_results = run_all_base_models(X_train, y_train)

    # ── 4. Optuna HPO ─────────────────────────────────────────────────────────
    if not args.quick:
        print(f"\n[4/7] Optuna HPO ({args.trials} trials each)...")
        best_params = run_optuna_hpo(X_train, y_train,
                                     n_trials_lgb=args.trials,
                                     n_trials_xgb=args.trials)
        lgb_best = best_params.get("LightGBM", LIGHTGBM_PARAMS)
        xgb_best = best_params.get("XGBoost",  XGBOOST_PARAMS)
    else:
        print("\n[4/7] Skipping HPO (--quick mode)")
        lgb_best, xgb_best = None, None

    # ── 5. Stacking ensemble ──────────────────────────────────────────────────
    print("\n[5/7] Training stacking ensemble...")
    ensemble, stack_metrics = train_stacking_ensemble(
        X_train, y_train, lgb_params=lgb_best, xgb_params=xgb_best
    )
    joblib.dump(ensemble, MODELS_DIR / "stacking_ensemble.pkl")

    # ── 6. Final evaluation on hold-out ──────────────────────────────────────
    print("\n[6/7] Evaluating on hold-out test set...")
    test_scores = ensemble.predict_proba(X_test)[:, 1]
    threshold   = get_optimal_threshold(y_test, test_scores)
    y_pred      = (test_scores >= threshold).astype(int)

    final_metrics = compute_all_metrics(y_test, test_scores, threshold)
    print("\n  Final metrics (hold-out):")
    for k, v in final_metrics.items():
        print(f"    {k:15s}: {v}")

    # Collect OOF scores for all models + ensemble for comparative plots
    all_oof_scores = {name: m["oof_probs"] for name, m in cv_results.items()}
    all_oof_scores["StackingEnsemble"] = stack_metrics["oof_probs"]

    # Comparative plots (OOF predictions on train set)
    print("\n  Generating comparison plots...")
    plot_roc_curves(all_oof_scores, y_train,
                    str(REPORTS_DIR / "roc_curves.png"))
    plot_pr_curves(all_oof_scores, y_train,
                   str(REPORTS_DIR / "pr_curves.png"))
    plot_calibration(all_oof_scores, y_train,
                     str(REPORTS_DIR / "calibration.png"))

    # Hold-out plots
    plot_ks_chart(y_test, test_scores, "StackingEnsemble",
                  str(REPORTS_DIR / "ks_chart.png"))
    plot_confusion_matrix(y_test, y_pred, "StackingEnsemble",
                          str(REPORTS_DIR / "confusion_matrix.png"))

    # ── 7. SHAP + Fairness ────────────────────────────────────────────────────
    print("\n[7/7] Computing SHAP values and fairness analysis...")

    # Use the best single model for SHAP (LightGBM inside ensemble, or standalone)
    from src.models import get_base_models
    from imblearn.combine import SMOTEENN
    lgb_model = get_base_models()["LightGBM"]
    smoteenn  = SMOTEENN(random_state=42)
    try:
        X_res, y_res = smoteenn.fit_resample(X_train, y_train)
    except Exception:
        X_res, y_res = X_train, y_train
    lgb_model.fit(X_res, y_res)
    joblib.dump(lgb_model, MODELS_DIR / "lightgbm_best.pkl")

    shap_vals, _, X_sample = compute_shap_values(lgb_model, X_test, feature_names)
    plot_shap_summary(shap_vals, X_sample, feature_names,
                      str(REPORTS_DIR / "shap_summary.png"))
    plot_shap_dependence(shap_vals, X_sample, feature_names, top_n=3,
                         save_path_prefix=str(REPORTS_DIR / "shap_dep"))

    # Fairness
    fairness_analysis(
        X_test_raw.reset_index(drop=True), y_test, test_scores,
        group_col="A1",
        threshold=threshold,
        save_path=str(REPORTS_DIR / "fairness_analysis.csv"),
    )

    # Results table
    all_cv = {**cv_results, "StackingEnsemble": stack_metrics}
    results_df = build_results_table(all_cv)
    print("\n  Model comparison (CV metrics):")
    print(results_df.to_string(index=False))
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Models   → {MODELS_DIR}")
    print(f"  Reports  → {REPORTS_DIR}")
    ens_auc = stack_metrics["auc_mean"]
    ens_gini = stack_metrics["gini_mean"]
    ens_ks   = stack_metrics["ks_mean"]
    print(f"\n  Stacking Ensemble (10-fold OOF):")
    print(f"    AUC-ROC : {ens_auc:.4f}  {'✓' if ens_auc >= 0.87 else '✗'} (target ≥ 0.87)")
    print(f"    Gini    : {ens_gini:.4f}  {'✓' if ens_gini >= 0.74 else '✗'} (target ≥ 0.74)")
    print(f"    KS      : {ens_ks:.4f}  {'✓' if ens_ks >= 0.50 else '✗'} (target ≥ 0.50)")
    print("=" * 60)


if __name__ == "__main__":
    main()
