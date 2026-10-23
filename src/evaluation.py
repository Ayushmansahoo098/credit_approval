"""
Evaluation module:
  - Full metrics suite (AUC, Gini, KS, PR-AUC, Brier, ECE)
  - ROC, PR, calibration, KS charts
  - SHAP global + dependence + waterfall
  - Fairness analysis (Disparate Impact, equalized odds)
  - Confusion matrix at optimal threshold
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

from src.utils import ks_statistic, gini_coefficient, get_optimal_threshold

REPORTS_DIR = Path("outputs/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Metric computation ────────────────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray, y_score: np.ndarray,
                        threshold: float | None = None) -> dict:
    """Compute full metrics suite."""
    auc = roc_auc_score(y_true, y_score)
    if threshold is None:
        threshold = get_optimal_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)

    # ECE (Expected Calibration Error)
    fraction_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10)
    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))

    return {
        "AUC_ROC":  round(auc, 4),
        "Gini":     round(gini_coefficient(auc), 4),
        "KS":       round(ks_statistic(y_true, y_score), 4),
        "PR_AUC":   round(average_precision_score(y_true, y_score), 4),
        "Brier":    round(brier_score_loss(y_true, y_score), 4),
        "ECE":      round(ece, 4),
        "Threshold": round(threshold, 4),
        "Precision": round(float(confusion_matrix(y_true, y_pred)[1, 1] /
                               (confusion_matrix(y_true, y_pred)[:, 1].sum() + 1e-9)), 4),
        "Recall":   round(float(confusion_matrix(y_true, y_pred)[1, 1] /
                               (confusion_matrix(y_true, y_pred)[1, :].sum() + 1e-9)), 4),
    }


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_roc_curves(models_scores: dict, y_true: np.ndarray,
                    save_path: str | None = None):
    """Overlay ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in models_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = save_path or str(REPORTS_DIR / "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_pr_curves(models_scores: dict, y_true: np.ndarray,
                   save_path: str | None = None):
    """Overlay Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, scores in models_scores.items():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        ax.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = save_path or str(REPORTS_DIR / "pr_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_calibration(models_scores: dict, y_true: np.ndarray,
                     save_path: str | None = None):
    """Calibration plot for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, scores in models_scores.items():
        fraction_pos, mean_pred = calibration_curve(y_true, scores, n_bins=10)
        ax.plot(mean_pred, fraction_pos, marker="o", label=name)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves")
    ax.legend()
    plt.tight_layout()
    path = save_path or str(REPORTS_DIR / "calibration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_ks_chart(y_true: np.ndarray, y_score: np.ndarray,
                  model_name: str = "Best Model",
                  save_path: str | None = None):
    """KS chart showing separation of approval/rejection distributions."""
    df = pd.DataFrame({"score": y_score, "target": y_true}).sort_values("score")
    n = len(df)
    neg_cum = (1 - df["target"]).cumsum() / (1 - df["target"]).sum()
    pos_cum = df["target"].cumsum() / df["target"].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(n), neg_cum.values, label="Rejected (-)  CDF")
    ax.plot(range(n), pos_cum.values, label="Approved (+) CDF")
    ks = float(np.max(np.abs(neg_cum.values - pos_cum.values)))
    ax.set_title(f"KS Chart — {model_name}  (KS={ks:.3f})")
    ax.set_xlabel("Sample (sorted by score)")
    ax.set_ylabel("Cumulative Rate")
    ax.legend()
    plt.tight_layout()
    path = save_path or str(REPORTS_DIR / "ks_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "Best Model",
                          save_path: str | None = None):
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = save_path or str(REPORTS_DIR / "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ── SHAP ─────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X: np.ndarray, feature_names: list[str] | None = None,
                        max_samples: int = 300):
    """Compute SHAP values using TreeExplainer."""
    import shap
    X_sample = X[:max_samples] if len(X) > max_samples else X
    # Try TreeExplainer first (tree models), fall back to KernelExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        # For binary classification some models return list; take class-1
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    except Exception:
        import shap
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_vals = explainer.shap_values(X_sample)[:, :, 1]

    return shap_vals, explainer, X_sample


def plot_shap_summary(shap_values, X_sample: np.ndarray,
                      feature_names: list[str] | None = None,
                      save_path: str | None = None):
    import shap
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.tight_layout()
    path = save_path or str(REPORTS_DIR / "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


def plot_shap_dependence(shap_values, X_sample: np.ndarray,
                         feature_names: list[str] | None = None,
                         top_n: int = 3,
                         save_path_prefix: str | None = None):
    """Plot SHAP dependence plots for top N features."""
    import shap
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    paths = []
    for rank, idx in enumerate(top_idx):
        fname = feature_names[idx] if feature_names else f"feature_{idx}"
        # Sanitize fname for filesystem
        fname_safe = fname.replace("/", "_").replace(" ", "_")
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(idx, shap_values, X_sample,
                             feature_names=feature_names, ax=ax, show=False)
        plt.tight_layout()
        prefix = save_path_prefix or str(REPORTS_DIR / "shap_dep")
        path = f"{prefix}_{rank+1}_{fname_safe}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        paths.append(path)
    return paths


# ── Fairness analysis ─────────────────────────────────────────────────────────

def fairness_analysis(
    df_raw: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_col: str = "A1",
    threshold: float | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Compute fairness metrics by group (A1 — binary attribute).
    Returns DataFrame with per-group:
      - Approval rate
      - TPR (True Positive Rate)
      - FPR (False Positive Rate)
      - Disparate Impact Ratio
      - AUC
    """
    if threshold is None:
        threshold = get_optimal_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    group = df_raw[group_col].values
    unique_groups = [g for g in np.unique(group) if pd.notna(g)]

    rows = []
    for g in unique_groups:
        mask = group == g
        n = mask.sum()
        if n == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        ys = y_score[mask]

        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        approval_rate = yp.mean()
        auc = roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else float("nan")
        rows.append({
            "Group": g,
            "N": int(n),
            "Approval_Rate": round(approval_rate, 4),
            "TPR": round(tpr, 4),
            "FPR": round(fpr, 4),
            "AUC": round(auc, 4) if not np.isnan(auc) else "N/A",
        })

    result = pd.DataFrame(rows)

    # Disparate Impact Ratio (reference = group with highest approval rate)
    if len(result) >= 2:
        max_rate = result["Approval_Rate"].max()
        result["Disparate_Impact"] = (result["Approval_Rate"] / (max_rate + 1e-9)).round(4)
        result["Passes_80pct_Rule"] = result["Disparate_Impact"] >= 0.8

    path = save_path or str(REPORTS_DIR / "fairness_analysis.csv")
    result.to_csv(path, index=False)
    print(f"  Fairness report saved → {path}")
    print(result.to_string(index=False))
    return result


# ── Summary report ────────────────────────────────────────────────────────────

def build_results_table(all_metrics: dict) -> pd.DataFrame:
    """Assemble a comparison DataFrame from per-model metric dicts."""
    rows = []
    for model_name, m in all_metrics.items():
        rows.append({
            "Model": model_name,
            "AUC_ROC":  m.get("auc_mean", m.get("AUC_ROC", "?")),
            "Gini":     m.get("gini_mean", m.get("Gini", "?")),
            "KS":       m.get("ks_mean",   m.get("KS", "?")),
            "PR_AUC":   m.get("PR_AUC", "?"),
            "Brier":    m.get("Brier", "?"),
            "ECE":      m.get("ECE", "?"),
        })
    df = pd.DataFrame(rows).sort_values("AUC_ROC", ascending=False)
    path = str(REPORTS_DIR / "model_comparison.csv")
    df.to_csv(path, index=False)
    return df
