"""
Training orchestration:
  - Stratified 10-fold CV with SMOTEENN applied inside each fold
  - Optuna HPO for LightGBM and XGBoost
  - MLflow logging of params + metrics
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.combine import SMOTEENN

from src.utils import RANDOM_STATE, ks_statistic, gini_coefficient
from src.models import (
    get_base_models, get_stacking_ensemble,
    lgb_optuna_space, xgb_optuna_space,
    LIGHTGBM_PARAMS, XGBOOST_PARAMS,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── CV evaluation ─────────────────────────────────────────────────────────────

def evaluate_model_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    apply_smoteenn: bool = True,
    model_name: str = "model",
) -> dict:
    """
    Runs StratifiedKFold CV. For each fold:
      1. Fit SMOTEENN on training split only
      2. Fit model on resampled data
      3. Predict on original (unaugmented) validation split
    Returns dict of metric arrays across folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {"auc": [], "gini": [], "ks": []}
    oof_probs = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if apply_smoteenn:
            smoteenn = SMOTEENN(random_state=RANDOM_STATE)
            try:
                X_tr, y_tr = smoteenn.fit_resample(X_tr, y_tr)
            except Exception:
                pass  # fallback: use original if SMOTEENN fails

        import sklearn.base
        m = sklearn.base.clone(model)
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs

        auc = roc_auc_score(y_val, probs)
        fold_metrics["auc"].append(auc)
        fold_metrics["gini"].append(gini_coefficient(auc))
        fold_metrics["ks"].append(ks_statistic(y_val, probs))

    fold_metrics["oof_probs"] = oof_probs
    fold_metrics["auc_mean"] = float(np.mean(fold_metrics["auc"]))
    fold_metrics["auc_std"]  = float(np.std(fold_metrics["auc"]))
    fold_metrics["gini_mean"] = float(np.mean(fold_metrics["gini"]))
    fold_metrics["ks_mean"]   = float(np.mean(fold_metrics["ks"]))
    return fold_metrics


# ── MLflow wrapper ────────────────────────────────────────────────────────────

def run_all_base_models(
    X: np.ndarray,
    y: np.ndarray,
    experiment_name: str = "credit_approval_base",
) -> dict:
    """Train all base models with 10-fold CV and log to MLflow."""
    mlflow.set_experiment(experiment_name)
    results = {}
    models = get_base_models()

    for name, model in models.items():
        print(f"  Training {name}...")
        with mlflow.start_run(run_name=name):
            metrics = evaluate_model_cv(model, X, y, model_name=name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("cv_auc_mean", metrics["auc_mean"])
            mlflow.log_metric("cv_auc_std",  metrics["auc_std"])
            mlflow.log_metric("cv_gini_mean", metrics["gini_mean"])
            mlflow.log_metric("cv_ks_mean",   metrics["ks_mean"])
            results[name] = metrics
            print(f"    AUC={metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}  "
                  f"Gini={metrics['gini_mean']:.4f}  KS={metrics['ks_mean']:.4f}")
    return results


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def _make_objective(model_fn, space_fn, X, y):
    """Factory for Optuna objective function."""
    def objective(trial):
        params = space_fn(trial)
        model = model_fn(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        aucs = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            smoteenn = SMOTEENN(random_state=RANDOM_STATE)
            try:
                X_tr, y_tr = smoteenn.fit_resample(X_tr, y_tr)
            except Exception:
                pass
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, probs))
        return float(np.mean(aucs))
    return objective


def run_optuna_hpo(
    X: np.ndarray,
    y: np.ndarray,
    n_trials_lgb: int = 60,
    n_trials_xgb: int = 60,
    experiment_name: str = "credit_approval_hpo",
) -> dict:
    """
    Run Optuna HPO for LightGBM and XGBoost.
    Returns best params for each.
    """
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    best_params = {}

    # LightGBM
    print(f"  Optuna LightGBM ({n_trials_lgb} trials)...")
    lgb_study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    lgb_study.optimize(
        _make_objective(LGBMClassifier, lgb_optuna_space, X, y),
        n_trials=n_trials_lgb,
        show_progress_bar=False,
    )
    best_params["LightGBM"] = lgb_study.best_params
    print(f"    Best LGB AUC: {lgb_study.best_value:.4f}")

    # XGBoost
    print(f"  Optuna XGBoost ({n_trials_xgb} trials)...")
    xgb_study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    xgb_study.optimize(
        _make_objective(XGBClassifier, xgb_optuna_space, X, y),
        n_trials=n_trials_xgb,
        show_progress_bar=False,
    )
    best_params["XGBoost"] = xgb_study.best_params
    print(f"    Best XGB AUC: {xgb_study.best_value:.4f}")

    # Log best to MLflow
    mlflow.set_experiment(experiment_name)
    for model_name, params in best_params.items():
        with mlflow.start_run(run_name=f"best_{model_name}"):
            mlflow.log_params(params)

    return best_params


# ── Stacking ensemble training ────────────────────────────────────────────────

def train_stacking_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    lgb_params: dict | None = None,
    xgb_params: dict | None = None,
    experiment_name: str = "credit_approval_stacking",
) -> tuple:
    """
    Train stacking ensemble with 10-fold OOF evaluation.
    Returns (fitted_model, metrics_dict).
    """
    print("  Training Stacking Ensemble...")
    mlflow.set_experiment(experiment_name)

    ensemble = get_stacking_ensemble(lgb_params=lgb_params, xgb_params=xgb_params)
    metrics = evaluate_model_cv(ensemble, X, y, model_name="StackingEnsemble")

    with mlflow.start_run(run_name="StackingEnsemble"):
        mlflow.log_metric("cv_auc_mean",  metrics["auc_mean"])
        mlflow.log_metric("cv_gini_mean", metrics["gini_mean"])
        mlflow.log_metric("cv_ks_mean",   metrics["ks_mean"])

    # Refit on full data
    smoteenn = SMOTEENN(random_state=RANDOM_STATE)
    try:
        X_res, y_res = smoteenn.fit_resample(X, y)
    except Exception:
        X_res, y_res = X, y

    ensemble_final = get_stacking_ensemble(lgb_params=lgb_params, xgb_params=xgb_params)
    ensemble_final.fit(X_res, y_res)

    print(f"    Stacking AUC={metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}  "
          f"Gini={metrics['gini_mean']:.4f}  KS={metrics['ks_mean']:.4f}")
    return ensemble_final, metrics
