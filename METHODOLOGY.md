# Credit Approval ML — Methodology & Architecture

This document describes the modeling architecture, validation methodology, finance-specific evaluation metrics, and regulatory fairness framework implemented in this repository.

---

## 1. Architectural Overview

The Credit Approval system is organized into a modular, production-grade machine learning architecture designed to prevent data leakage, optimize finance-specific scorecard metrics, satisfy regulatory compliance (EEOC 80% rule), and serve real-time credit decisioning.

### 1.1 End-to-End Pipeline & System Flow

```mermaid
flowchart TD
    subgraph S1["Stage 1: Ingestion & Stratified Split"]
        RAW["Raw Dataset (crx.data)<br/>690 applicants • 15 features"] --> SPLIT{"Stratified Hold-out Split<br/>(85% Train / 15% Test)"}
        SPLIT -->|"586 samples"| TRAIN_RAW["Development Set (X_tr_raw, y_train)"]
        SPLIT -->|"104 samples"| TEST_RAW["Hold-out Test Set (X_test_raw, y_test)"]
    end

    subgraph S2["Stage 2: Preprocessing & Feature Engineering"]
        TRAIN_RAW --> PIPE["Preprocessing Pipeline<br/>(Fit on Train Partition)"]
        PIPE --> IND["1. Missing Value Indicators (A1-A14)"]
        IND --> IMP["2. Imputation (Mode for Cat, Median for Num)"]
        IMP --> FE["3. Feature Engineering (Log1p, Ratios, Interactions)"]
        FE --> ENC["4. Encoding (Binary Labels + OneHot for Multi)"]
        ENC --> SCL["5. Scaling (StandardScaler on continuous)"]
        SCL --> X_TRAIN["Transformed Train Matrix<br/>(586 samples × 53 features)"]
        
        TEST_RAW -.->|"transform() only<br/>(Zero Leakage)"| PIPE
        PIPE -.-> X_TEST["Transformed Test Matrix<br/>(104 samples × 53 features)"]
    end

    subgraph S3["Stage 3: Cross-Validation & Leakage-Free Resampling"]
        X_TRAIN --> CV_SPLIT["10-Fold StratifiedKFold Loop"]
        CV_SPLIT --> FOLD_TR["Fold Train (90% ~527 samples)"]
        CV_SPLIT --> FOLD_VAL["Fold Validation (10% ~59 samples)"]
        FOLD_TR --> SMOTE["SMOTEENN Resampling<br/>(Applied STRICTLY inside fold)"]
        SMOTE --> RESAMPLED["Balanced Fold Training Data"]
    end

    subgraph S4["Stage 4: Model Diversity & Stacking Ensemble"]
        RESAMPLED --> M1["Logistic Regression<br/>(Linear Benchmark)"]
        RESAMPLED --> M2["Random Forest<br/>(Bagging Ensemble)"]
        RESAMPLED --> M3["XGBoost<br/>(Gradient Boosting)"]
        RESAMPLED --> M4["LightGBM<br/>(Leaf-wise Tree GBDT)"]
        RESAMPLED --> M5["CatBoost<br/>(Symmetric Oblivious Trees)"]
        
        M1 & M2 & M3 & M4 & M5 --> OOF["Generate Out-of-Fold (OOF) Probabilities"]
        FOLD_VAL -.->|"Unaugmented Evaluation"| OOF
        
        OOF --> STACK["Stacking Classifier<br/>(Meta-Learner: Logistic Regression)"]
        STACK --> MODEL_PKL["Serialized Model Artifacts<br/>(stacking_ensemble.pkl, preprocessor.pkl)"]
    end

    subgraph S5["Stage 5: Verification, Explainability & Serving"]
        X_TEST --> EVAL["Hold-out Evaluation<br/>AUC: 0.9659 • KS: 0.8786 • Gini: 0.9318"]
        X_TEST --> SHAP_EX["SHAP TreeExplainer<br/>(Global Summary + Dependence)"]
        TEST_RAW --> FAIR["Fairness Audit Engine<br/>(DIR = 84.38% • Passes 80% Rule)"]
        
        MODEL_PKL --> APP["CreditIQ Web Portal & REST API<br/>(Flask Engine on Port 8080)"]
        APP --> DECISION["Real-time Underwriting Decision<br/>Approved/Rejected • Risk Tier • FCRA Adverse Action"]
    end
```

---

### 1.2 Stacking Ensemble Architecture

The production scoring engine uses a two-tier stacking architecture that combines gradient boosting models with diverse splitting schemes, blended by an L2-regularized meta-learner:

```mermaid
flowchart LR
    INPUT["53 Transformed Input Features"] --> L0_CAT["Level-0 Base Learner 1:<br/>CatBoost Classifier"]
    INPUT --> L0_LGB["Level-0 Base Learner 2:<br/>LightGBM Classifier"]
    INPUT --> L0_XGB["Level-0 Base Learner 3:<br/>XGBoost Classifier"]

    L0_CAT -->|"P(Approved) CatBoost"| BLEND["Out-of-Fold Probability Vector<br/>[p_cat, p_lgb, p_xgb]"]
    L0_LGB -->|"P(Approved) LightGBM"| BLEND
    L0_XGB -->|"P(Approved) XGBoost"| BLEND

    BLEND --> L1_META["Level-1 Meta-Learner:<br/>Logistic Regression (L2 Penalty)"]
    L1_META --> OUT["Calibrated Approval Probability<br/>P(Approved | X)"]
    OUT --> CUTOFF{"Optimal Cutoff Decision<br/>Threshold = 0.1524"}
    CUTOFF -->|"P ≥ 0.1524"| APP["APPROVED (Prime / Moderate Risk)"]
    CUTOFF -->|"P < 0.1524"| REJ["REJECTED (Subprime / High Risk)"]
```

---

### 1.3 Subsystem Responsibility Matrix

| Subsystem Component | Module Location | Core Responsibility | Leakage Protection & Invariants |
| :--- | :--- | :--- | :--- |
| **Data Ingestion** | [`src/utils.py`](src/utils.py) | Ingests `crx.data`, maps column schema ($A_1$–$A_{16}$), converts targets (`+` $\rightarrow$ 1, `-` $\rightarrow$ 0). | Stratified split isolated before any transformation. |
| **Feature Transformation** | [`src/preprocessing.py`](src/preprocessing.py) | Adds missingness flags, medoid/mode imputation, log-scaling ($A_{11}$, $A_{15}$), interactions ($A_2 \times A_3$), and OHE. | Fitted strictly on development split; hold-out and test data only call `transform()`. |
| **Resampling Orchestration** | [`src/training.py`](src/training.py) | Combines SMOTE oversampling with ENN undersampling to clean noisy boundaries. | **Executed strictly within each CV fold.** Never exposed to validation folds. |
| **Model Cross-Validation** | [`src/training.py`](src/training.py) | Executes 10-fold Stratified CV, logs hyperparameter spaces and fold metrics to MLflow. | Out-of-fold predictions gathered across hold-out validation segments. |
| **Stacking Meta-Learner** | [`src/models.py`](src/models.py) | Fits Logistic Regression over base model probability vectors to reduce decision variance. | Trained exclusively on cross-validated OOF predictions. |
| **Risk & Fairness Audit** | [`src/evaluation.py`](src/evaluation.py) | Calculates finance metrics (Gini, KS, Brier, ECE), computes SHAP values, and evaluates Disparate Impact Ratio. | Evaluated on hold-out test set unseen during training. |
| **Interactive Serving** | [`app.py`](app.py) | Exposes REST API and Web UI for real-time applicant evaluation and FCRA reason code extraction. | Consumes serialized production pipeline (`preprocessor.pkl`, `stacking_ensemble.pkl`). |

---

## 2. Preventing Data Leakage

A critical failure mode in credit risk modeling is applying resampling or transformations prior to cross-validation splits:

1. **Resampling Leakage:** Applying SMOTE or SMOTEENN across the entire dataset introduces synthetic data points informed by the test/validation distribution. In this project, `SMOTEENN` is initialized and fitted **strictly inside each training fold**. The validation fold remains untouched and reflects the true production population distribution.
2. **Preprocessing Leakage:** Imputers and scalers (`SimpleImputer`, `StandardScaler`) are fit exclusively on the training partition (`X_tr_raw`) and applied via `transform()` to the hold-out set (`X_test_raw`).

---

## 3. Modeling Strategy & Ensembling

### Base Learners
- **Logistic Regression:** Serves as a transparent linear benchmark. Uses L2 regularization and scaled features.
- **Random Forest:** Non-linear bagging model with 200 estimators.
- **Gradient Boosted Trees (XGBoost, LightGBM, CatBoost):** Captures high-order feature interactions. CatBoost natively handles categorical combinations with symmetric trees, minimizing variance.

### Stacking Ensemble
- **Architecture:** Stacking classifier combining the top-3 gradient boosting architectures (`CatBoost`, `LightGBM`, `XGBoost`).
- **Meta-Learner:** L2-regularized `LogisticRegression` that maps the Out-of-Fold probability outputs of the base learners to a final calibrated risk score.
- **Benefit:** Reduces prediction variance and improves discrimination on borderline applications.

---

## 4. Banking & Credit Risk Metrics

Standard machine learning accuracy is unsuitable for credit approval due to class imbalance and asymmetric default risk. We report industry standards:

| Metric | Formula | Financial Interpretation |
|--------|---------|---------------------------|
| **AUC-ROC** | Area Under Receiver Operating Characteristic | Probability that a randomly selected approved applicant receives a higher score than a rejected applicant. Target: $\ge 0.87$. |
| **Gini Coefficient** | $\text{Gini} = 2 \times \text{AUC} - 1$ | The European banking standard for credit scorecards. Values $> 0.70$ represent investment-grade scorecard quality. Target: $\ge 0.74$. |
| **KS Statistic** | $\max_t \|F_{\text{approved}}(t) - F_{\text{rejected}}(t)\|$ | Maximum vertical separation between cumulative distribution functions. A KS $> 0.50$ is considered strong; $> 0.70$ is exceptional. Target: $\ge 0.50$. |
| **Brier Score** | $\frac{1}{N} \sum (p_i - y_i)^2$ | Strict calibration accuracy metric. $0$ is perfect; $0.25$ represents zero-skill guessing. |
| **Expected Calibration Error (ECE)** | Bin-weighted difference between predicted probability and actual frequency | Measures risk-based pricing reliability (e.g. interest rate sizing). |

---

## 5. Model Explainability (SHAP)

To satisfy consumer credit disclosure requirements (e.g., Fair Credit Reporting Act adverse action notices), model decisions must be explainable.

- **Framework:** SHAP (`TreeExplainer`) on the top-performing tree model.
- **Key Findings:**
  1. `A9` (Credit History / Prior Default flag) is the dominant driver of approval.
  2. `A11_log` (Active credit lines / Credit score proxy) provides steady monotonic approval gains.
  3. `A15_log` (Savings reserves) provides significant positive lift at higher thresholds.
  4. Missing value indicators contribute subtle negative adjustments, capturing non-random omissions.

---

## 6. Fairness & Algorithmic Bias Audit

Credit models must comply with fair lending standards (e.g., EEOC and Equal Credit Opportunity Act):

- **Protected Attribute Proxy:** Feature `A1` (Binary demographic group: `a` vs. `b`).
- **Disparate Impact Ratio (DIR):**
  $$\text{DIR} = \frac{\text{Approval Rate}(\text{Group } a)}{\text{Approval Rate}(\text{Group } b)}$$
- **Four-Fifths (80%) Rule:** If $\text{DIR} \ge 0.80$, the selection process has no adverse impact under standard regulatory scrutiny.
- **Current Audit Outcome:**
  - Group `a` Approval Rate: 37.5%
  - Group `b` Approval Rate: 44.4%
  - **DIR = 0.8438 (84.4%)** $\rightarrow$ **Passes the 80% Rule**.
