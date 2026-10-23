# Credit Approval ML — Operational Runbook

This guide covers environment management, command-line execution, MLflow experiment tracking, and notebook execution for the Credit Approval ML project.

---

## 1. Environment Setup

### System Prerequisites
- Python 3.11
- macOS Users: Install the OpenMP runtime for `xgboost` and `lightgbm`:
  ```bash
  brew install libomp
  ```

### Virtual Environment (Recommended with `uv`)
```bash
# 1. Create a Python 3.11 virtual environment
uv venv --python 3.11 .venv

# 2. Activate the environment
source .venv/bin/activate

# 3. Install project dependencies
uv pip install -r requirements.txt
```

*(Alternative using standard `python3` and `pip`):*
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Pipeline Execution

Activate the virtual environment first:
```bash
source .venv/bin/activate
```

### Quick Run (~1–2 Minutes)
Trains all base models (10-fold CV with SMOTEENN), trains the Stacking Ensemble, computes hold-out metrics, generates SHAP plots, and conducts the fairness audit. Skips Optuna HPO for fast local verification:
```bash
python main.py --quick
```

### Full Production Run (~10–15 Minutes)
Executes the full pipeline including automated Bayesian hyperparameter tuning via Optuna (60 trials per tree model):
```bash
python main.py
```

### Custom HPO Trial Budget
Specify the number of Optuna trials per model:
```bash
python main.py --trials 100
```

## 3. Interactive Web Application (CreditIQ)

A modern, high-precision credit underwriting dashboard is included:

```bash
# Start the web app on http://127.0.0.1:8080
source .venv/bin/activate
python app.py
```

Then open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in your web browser.

### Key Features
- **Applicant Persona Presets**: Quick-load *Prime*, *Borderline*, *High Risk (Default)*, or *Random* applications with 1 click.
- **Real-Time Underwriting Engine**: Scores applications through the preprocessor and Stacking Ensemble model.
- **Risk Classification & Decision**: Displays glowing Approved vs. Rejected banners, approval probability meter, and optimal decision threshold ($15.24\%$).
- **FCRA Adverse Action & SHAP Explanations**: Provides human-readable reason codes detailing exactly which factors helped or hurt the decision.
- **Executive Governance & Audit Tabs**: Embedded ROC curves, PR curves, calibration curves, confusion matrix, and EEOC 80% Rule disparate impact audit table.

---

## 4. Experiment Tracking with MLflow

All model runs, hyperparameter spaces, and cross-validation metrics are tracked in local MLflow runs (`mlruns/` or SQLite `mlflow.db`).

Launch the interactive UI:
```bash
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) in your browser to:
- Compare runs across base models (`credit_approval_base`) and ensembles (`credit_approval_stacking`).
- Inspect cross-validation metric distributions (`cv_auc_mean`, `cv_gini_mean`, `cv_ks_mean`).
- View logged model hyperparameters and artifacts.

---

## 4. Interactive Jupyter Notebooks

Three self-contained notebooks are provided under `notebooks/`:

```bash
jupyter notebook notebooks/
```

1. **`01_eda.ipynb`**: Exploratory data analysis, missing value maps, feature distributions, Cramér's V associations.
2. **`02_modeling.ipynb`**: Detailed step-by-step model building, SMOTEENN experiments, Optuna tuning.
3. **`03_evaluation.ipynb`**: Precision-Recall curves, calibration curves, SHAP interpretability, and disparate impact analysis.

---

## 5. Artifact Verification

Upon pipeline completion, check that the following files are populated in `outputs/`:

### Models (`outputs/models/`)
- `preprocessor.pkl`: Fitted scikit-learn preprocessing pipeline.
- `stacking_ensemble.pkl`: Fitted stacking classifier (CatBoost + LightGBM + XGBoost $\rightarrow$ Logistic Regression).
- `lightgbm_best.pkl`: Single best tree model for SHAP TreeExplainer.

### Reports & Visualizations (`outputs/reports/`)
- `model_comparison.csv`: Out-of-fold CV metrics for all tested models.
- `fairness_analysis.csv`: Disparate impact ratio and 80% rule results by demographic group.
- `roc_curves.png`: Comparative ROC curves across all algorithms.
- `pr_curves.png`: Precision-Recall curves.
- `calibration.png`: Probability calibration curves against ideal diagonal.
- `ks_chart.png`: Kolmogorov-Smirnov separation curve on the hold-out set.
- `confusion_matrix.png`: Evaluation at the F1-optimal decision threshold.
- `shap_summary.png`: Global feature importance summary.
- `shap_dep_*.png`: Partial dependence plots for top features (`A9`, `A11`, `A15`).
- `findings.md`: Comprehensive written analysis of model performance and governance findings.
