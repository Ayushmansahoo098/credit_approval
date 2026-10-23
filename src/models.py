"""
Model definitions and default hyperparameter configs.

Models:
  1. LogisticRegression  — interpretable baseline
  2. RandomForest        — ensemble baseline
  3. XGBoost             — industry standard
  4. LightGBM            — best tabular credit performance
  5. CatBoost            — native cat handling
  6. StackingEnsemble    — LGB + XGB + CatBoost → LR meta
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.utils import RANDOM_STATE


# ── Default configs ───────────────────────────────────────────────────────────

LOGISTIC_REGRESSION_PARAMS = dict(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

RANDOM_FOREST_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)

XGBOOST_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)

LIGHTGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

CATBOOST_PARAMS = dict(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=RANDOM_STATE,
    verbose=0,
    eval_metric="AUC",
)

# ── Model factory ─────────────────────────────────────────────────────────────

def get_base_models() -> dict:
    """Return dict of name → unfitted classifier."""
    return {
        "LogisticRegression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        "RandomForest": RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        "XGBoost": XGBClassifier(**XGBOOST_PARAMS),
        "LightGBM": LGBMClassifier(**LIGHTGBM_PARAMS),
        "CatBoost": CatBoostClassifier(**CATBOOST_PARAMS),
    }


def get_stacking_ensemble(
    lgb_params: dict | None = None,
    xgb_params: dict | None = None,
    cat_params: dict | None = None,
) -> StackingClassifier:
    """
    Stacking ensemble:
      Base learners: LightGBM, XGBoost, CatBoost
      Meta-learner:  Logistic Regression
    """
    lgb = LGBMClassifier(**(lgb_params or LIGHTGBM_PARAMS))
    xgb = XGBClassifier(**(xgb_params or XGBOOST_PARAMS))
    cat = CatBoostClassifier(**(cat_params or CATBOOST_PARAMS))

    estimators = [
        ("lgb", lgb),
        ("xgb", xgb),
        ("cat", cat),
    ]
    meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                              random_state=RANDOM_STATE)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )


def calibrate_model(model, X_calib, y_calib, method: str = "sigmoid", cv: int = 5):
    """Wrap model in CalibratedClassifierCV and fit on calibration data."""
    cal = CalibratedClassifierCV(model, method=method, cv=cv)
    cal.fit(X_calib, y_calib)
    return cal


# ── Optuna search spaces ──────────────────────────────────────────────────────

def lgb_optuna_space(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def xgb_optuna_space(trial) -> dict:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        gamma=trial.suggest_float("gamma", 0, 5),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        eval_metric="auc",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
