# Credit Approval ML — Findings Report

**Dataset:** UCI Credit Approval — 690 applicants, 15 features, binary outcome (+/-)
**Evaluation:** 10-fold stratified cross-validation with SMOTEENN resampling inside each fold

---

## 1. Model Performance

All models were trained under identical conditions: SMOTEENN applied strictly inside each CV fold to prevent data leakage, and StratifiedKFold preserving the class ratio in every split.

| Model | AUC-ROC | Gini | KS |
|-------|---------|------|----|
| CatBoost | **0.9117** | **0.8234** | **0.7622** |
| Stacking Ensemble | 0.9094 | 0.8188 | 0.7516 |
| LightGBM | 0.9059 | 0.8119 | 0.7393 |
| XGBoost | 0.9038 | 0.8076 | 0.7495 |
| Logistic Regression | 0.9010 | 0.8020 | 0.7451 |
| Random Forest | 0.8918 | 0.7836 | 0.7292 |

**Key observation:** All models perform remarkably well, with even the simplest model (Logistic Regression, AUC 0.90) exceeding the project target of 0.87. This suggests the features in this dataset are highly informative — the credit decision boundary is largely learnable from the available attributes.

CatBoost edged out the stacking ensemble, likely because it handles the mixed categorical/continuous feature space particularly well natively. The stacking ensemble's marginal underperformance relative to CatBoost is not unusual: stacking adds value when base learners are diverse and individually strong, but can be hurt by the meta-learner's limited training data (only OOF predictions).

### What the metrics mean in credit context

- **AUC-ROC (0.91):** The model correctly ranks a randomly chosen approved applicant above a rejected one 91% of the time. Strong discrimination.
- **Gini (0.82):** The European banking standard equivalent (2×AUC−1). Values above 0.70 are generally considered production-ready for retail credit scorecards.
- **KS (0.76):** The maximum separation between the cumulative distributions of approved vs. rejected applicants. A KS above 0.50 is considered strong for credit scoring; 0.76 is exceptional.

---

## 2. Hold-Out Test Set Results (Stacking Ensemble)

On the 15% held-out test set (104 applicants not seen during training):

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9659 |
| Gini | 0.9318 |
| KS | 0.8786 |
| PR-AUC | 0.9303 |
| Brier Score | 0.0983 |
| Precision | 0.9545 |
| Recall | 0.9130 |
| Decision Threshold | 0.1524 |

The hold-out metrics are higher than the CV averages, which is partially a function of the small test set size (104 samples) and the variance inherent in such splits. The optimal decision threshold was 0.15 — lower than the default 0.50 — reflecting that the model's calibrated probabilities for this dataset concentrate in the 0.1–0.4 range, and the F1-optimal cut is well below the midpoint.

The **Brier score of 0.098** indicates well-calibrated probabilities (lower is better; 0 is perfect, 0.25 is the no-skill baseline for a balanced problem).

---

## 3. Feature Importance (SHAP)

SHAP (SHapley Additive exPlanations) analysis on the LightGBM model reveals which features most influence individual credit decisions:

**Top features by mean absolute SHAP value:**

1. **A9** — Binary attribute (likely employment status or prior default indicator). The single strongest predictor. A positive value strongly shifts the prediction toward approval.
2. **A11** — Continuous attribute (log-transformed). Likely a financial balance or credit history metric. Higher values increase approval probability.
3. **A15** — Continuous (log-transformed savings/assets). Higher values increase approval likelihood, consistent with financial capacity signaling.

**Interpretation:** The model is largely driven by a binary status flag (A9) and two financial magnitude features (A11, A15). This mirrors real-world credit scoring behavior, where employment status and financial reserves dominate decisions. The remaining features contribute smaller, incremental improvements.

The UCI dataset anonymizes column names deliberately, but the SHAP dependence plots (see `shap_dep_*.png`) show:
- A9: near-threshold effect — flips from strong negative to strong positive contribution at the binary boundary.
- A11 and A15: monotonically increasing SHAP contributions — more assets/credit history = higher approval probability.

---

## 4. Fairness Analysis

The binary attribute A1 (likely a gender or demographic proxy, anonymized in UCI) was used as the grouping variable.

| Group | N | Approval Rate | TPR | FPR | AUC | Disparate Impact | Passes 80% Rule |
|-------|---|---------------|-----|-----|-----|-----------------|-----------------|
| a | 32 | 37.5% | 84.6% | 5.3% | 0.935 | 0.844 | **Yes** |
| b | 72 | 44.4% | 93.9% | 2.6% | 0.981 | 1.000 | **Yes** |

**Disparate Impact Ratio (DIR):** Group a is approved at 84.4% the rate of group b. Under the EEOC 80% rule (a regulatory standard used in fair lending analysis), a DIR ≥ 0.80 is considered acceptable. Both groups pass this threshold.

**Equalized odds analysis:**
- Group a: TPR = 84.6%, FPR = 5.3%
- Group b: TPR = 93.9%, FPR = 2.6%

There is a meaningful difference in True Positive Rates (9.3 percentage points): qualified applicants in group a are approved less often than qualified applicants in group b. This gap is worth monitoring in a production setting, even though the DIR passes the 80% rule. A fairer threshold or post-processing calibration per group could reduce this disparity.

**Important caveat:** The UCI dataset anonymizes all sensitive attributes and the group labels (a/b) are not documented as gender. These results demonstrate the fairness evaluation methodology, not a confirmed finding about any real protected class.

---

## 5. Calibration

The model's predicted probabilities need to be reliable if they are used for risk-based pricing (e.g., setting interest rates based on default probability). A perfectly calibrated model predicts 70% approval probability for applicants who are approved 70% of the time.

The Brier score of 0.098 (hold-out) is well below the 0.25 no-skill baseline and indicates reasonable calibration. The calibration curves (see `calibration.png`) show the model's probabilities track the actual approval rates across deciles, with mild overconfidence at the extremes — a common pattern for boosting models.

For a production deployment, wrapping the final model in `CalibratedClassifierCV(method='sigmoid')` would tighten the calibration further.

---

## 6. What Drives Approvals — Summary

Based on SHAP and feature correlations from EDA:

| Signal | Direction | Strength |
|--------|-----------|----------|
| A9 (binary status flag) | Positive → approval | Very strong |
| A11 (financial metric, log-scaled) | Higher → approval | Strong |
| A15 (savings/assets, log-scaled) | Higher → approval | Strong |
| A4 / A5 (categorical attributes) | Varies by category | Moderate |
| A2 (likely age) | Older → slight approval lift | Weak–moderate |
| Missing value indicators | Missingness slightly negative | Weak |

The missing value indicators contributed small but non-zero SHAP values, confirming that refusal to provide certain information is itself a mild negative signal — consistent with credit scoring practice.

---

## 7. Conclusions

1. **The dataset is highly predictable.** Even logistic regression achieves AUC 0.90, suggesting the features encode credit risk information directly. A real-world dataset would be noisier.

2. **CatBoost is the best single model** for this mixed-type dataset, edging out the stacking ensemble by ~0.2% AUC. For production, CatBoost's native categorical handling reduces preprocessing complexity.

3. **The stacking ensemble adds robustness over any single model**, with lower variance (std 0.033 vs. 0.035 for CatBoost), which matters for regulatory model validation requirements.

4. **Fairness is borderline.** Group a applicants have a 9-point lower TPR than group b. While the DIR passes the 80% rule, a production system would investigate further and consider group-specific thresholds.

5. **Calibration is good but improvable.** Post-hoc calibration via Platt scaling would be recommended before using model outputs for risk-based pricing.

6. **SMOTEENN + inside-CV resampling was critical.** Preliminary experiments applying resampling before the CV split produced inflated AUC estimates (~0.02–0.03 higher) — a common portfolio mistake this project avoids.
