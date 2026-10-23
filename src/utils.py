"""Utility functions for the credit approval ML project."""

import numpy as np
import pandas as pd


COLUMN_NAMES = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10",
                "A11","A12","A13","A14","A15","A16"]

CATEGORICAL_COLS = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
CONTINUOUS_COLS  = ["A2", "A3", "A8", "A11", "A14", "A15"]

BINARY_CATS  = ["A1", "A9", "A10", "A12"]
MULTI_CATS   = ["A4", "A5", "A6", "A7", "A13"]
SKEWED_COLS  = ["A15", "A11"]

TARGET_COL = "A16"
RANDOM_STATE = 42


def load_raw_data(path: str) -> pd.DataFrame:
    """Load crx.data, assign column names, replace '?' with NaN, encode target."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    df[TARGET_COL] = df[TARGET_COL].map({"+": 1, "-": 0})
    return df


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association between two categorical Series."""
    from scipy.stats import chi2_contingency
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2corr / denom)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS statistic: max separation between TPR and FPR curves."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def gini_coefficient(auc: float) -> float:
    """Gini = 2 * AUC - 1."""
    return 2 * auc - 1


def get_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray,
                          metric: str = "f1") -> float:
    """Find threshold that maximises F1 (or minimises approval rate disparity)."""
    from sklearn.metrics import f1_score, precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])
