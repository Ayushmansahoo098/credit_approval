# Credit Approval Dataset — Data Dictionary

This document details the features, distributions, missing value patterns, and business interpretations for the UCI Credit Approval dataset (`crx.data`).

---

## Overview

- **Source:** [UCI Machine Learning Repository — Credit Approval Data Set](https://archive.ics.uci.edu/ml/datasets/Credit+Approval)
- **Total Records:** 690 applicants
- **Feature Count:** 15 input features (8 categorical, 7 numerical) + 1 binary target
- **Target Distribution:**
  - `+` (Approved, 1): 307 applicants (44.5%)
  - `-` (Rejected, 0): 383 applicants (55.5%)

All original feature names were anonymized by the data contributors to preserve bank customer privacy. Below are the structural properties and empirical business interpretations.

---

## Feature Specifications

| Feature | Type | Distinct Values / Range | Missing | Domain Interpretation & Notes |
|---------|------|-------------------------|---------|--------------------------------|
| **A1**  | Categorical (Binary) | `a`, `b` | 12 (1.7%) | Demographic proxy (e.g., Sex / Marital status). Used for disparate impact fairness evaluation. |
| **A2**  | Continuous | 13.75 – 80.25 | 12 (1.7%) | Applicant Age (years). Imputed with median. |
| **A3**  | Continuous | 0.00 – 28.00 | 0 (0.0%) | Debt / Income or Expense metric. |
| **A4**  | Categorical | `u`, `y`, `l`, `t` | 6 (0.9%) | Customer Bank Account / Relationship Status (e.g., Checking/Savings holder). |
| **A5**  | Categorical | `g`, `p`, `gg` | 6 (0.9%) | Bank customer type / segment. Highly correlated with A4. |
| **A6**  | Categorical | 14 distinct codes (`c`, `d`, `cc`, `i`, `j`, `k`, `m`, `r`, `q`, `w`, `x`, `e`, `aa`, `ff`) | 9 (1.3%) | Employment sector or Occupation category. One-hot encoded. |
| **A7**  | Categorical | 9 distinct codes (`v`, `h`, `bb`, `j`, `n`, `z`, `dd`, `ff`, `o`) | 9 (1.3%) | Ethnicity or Residential stability indicator. One-hot encoded. |
| **A8**  | Continuous | 0.00 – 28.50 | 0 (0.0%) | Years Employed / Time at current residence. |
| **A9**  | Categorical (Binary) | `t` (True), `f` (False) | 0 (0.0%) | Prior Default / Credit History clean flag. Single strongest predictor of approval. |
| **A10** | Categorical (Binary) | `t` (True), `f` (False) | 0 (0.0%) | Employment Status (Currently employed flag). |
| **A11** | Continuous | 0 – 67 | 0 (0.0%) | Credit Score / Number of active credit lines. Heavy right skew; log1p-transformed. |
| **A12** | Categorical (Binary) | `t` (True), `f` (False) | 0 (0.0%) | Driver's License / Identity verification flag. |
| **A13** | Categorical | `g` (by birth), `p`, `s` | 0 (0.0%) | Citizenship / Residency status. |
| **A14** | Continuous | 00000 – 02000 | 13 (1.9%) | Monthly payment / Zip code / Credit limit scale. Imputed with median. |
| **A15** | Continuous | 0 – 100,000 | 0 (0.0%) | Liquid savings or Annual income. Heavy right skew; log1p-transformed. |
| **A16** | Target (Binary) | `+` (1), `-` (0) | 0 (0.0%) | Credit decision: Approved (`+` -> 1) vs. Rejected (`-` -> 0). |

---

## Missing Value Strategy

Missingness in retail credit is rarely Missing Completely at Random (MCAR). Applicants who fail or decline to disclose certain financial indicators (MNAR) carry predictive risk.

1. **Indicator Flags:** Explicit binary columns (`A1_missing`, `A2_missing`, `A4_missing`, `A5_missing`, `A6_missing`, `A7_missing`, `A14_missing`) are added by `MissingIndicatorAdder`.
2. **Imputation:**
   - Categorical columns are imputed with the most frequent value (mode).
   - Numerical columns are imputed with the median.
   - All imputers are fitted strictly on the training partition to prevent data leakage.

---

## Engineered Features

The pipeline generates three derived domain features in `FeatureEngineer`:

1. **Log-transformed variables (`A11_log`, `A15_log`):**
   $$\text{feature\_log} = \ln(1 + x)$$
   Compresses heavy right-tailed financial magnitude distributions into normal-like bell curves.

2. **Age × Income Proxy Interaction (`A2_x_A3`):**
   $$A2\_x\_A3 = A2 \times A3$$
   Captures earning stage and debt-carrying capacity across applicant life stages.

3. **Savings-to-Payment Ratio (`savings_to_payment`):**
   $$\text{savings\_to\_payment} = \frac{A15}{A14 + 1}$$
   Measures liquidity buffer relative to obligations, a core metric in credit underwriting.
