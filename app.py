"""
Credit Approval ML — Web Application Backend
Serves interactive underwriting dashboard, real-time risk scoring,
SHAP adverse action explanations, and model governance reports.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

from src.utils import load_raw_data, TARGET_COL, COLUMN_NAMES
from src.preprocessing import get_feature_names

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "raw" / "crx.data"
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load Model Artifacts ──────────────────────────────────────────────────────
print("Loading preprocessor and models...")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
ensemble = joblib.load(MODELS_DIR / "stacking_ensemble.pkl")
lgb_model = joblib.load(MODELS_DIR / "lightgbm_best.pkl")
feature_names = get_feature_names(preprocessor)

# Load raw data for preset generation
raw_df = load_raw_data(str(DATA_PATH))

# Initialize SHAP explainer
import shap
explainer = shap.TreeExplainer(lgb_model)
print("All artifacts successfully loaded into memory.")

OPTIMAL_THRESHOLD = 0.1524  # F1-optimal threshold from hold-out evaluation

# Domain mappings for human-readable labels
FEATURE_LABELS = {
    "A1": "Demographic Group (Proxy)",
    "A2": "Applicant Age",
    "A3": "Debt / Income Metric",
    "A4": "Relationship / Account Status",
    "A5": "Customer Segment",
    "A6": "Employment / Occupation Sector",
    "A7": "Ethnicity / Residential Stability",
    "A8": "Years Employed / Time at Job",
    "A9": "Clean Prior Credit (No Default)",
    "A10": "Current Employment Status",
    "A11": "Active Credit Lines / Score",
    "A12": "Driver's License / ID Verified",
    "A13": "Citizenship / Residency Status",
    "A14": "Monthly Payment / Zip Code Scale",
    "A15": "Liquid Savings / Assets ($)",
}


def explain_shap_feature(feat_name: str, val: float, shap_val: float) -> str:
    """Generate human-readable FCRA adverse action or approval reasoning."""
    prefix = feat_name.split("_")[0]
    direction = "Supports approval" if shap_val > 0 else "Increases credit risk"
    
    if prefix == "A9":
        return f"{'Strong clean credit history' if shap_val > 0 else 'History of prior default / delinquencies'} ({direction})"
    elif prefix == "A15":
        return f"{'High asset / savings reserves' if shap_val > 0 else 'Limited liquid savings reserves'} ({direction})"
    elif prefix == "A11":
        return f"{'Favorable credit depth / active trade lines' if shap_val > 0 else 'Thin credit file or low activity'} ({direction})"
    elif prefix == "A8":
        return f"{'Longer employment tenure' if shap_val > 0 else 'Short employment tenure'} ({direction})"
    elif prefix == "A3":
        return f"{'Favorable debt-to-income metric' if shap_val > 0 else 'Elevated debt obligations'} ({direction})"
    elif prefix == "A2":
        return f"Applicant age & earning stability ({direction})"
    elif "missing" in feat_name:
        return f"Missing disclosure in {prefix} ({direction})"
    else:
        return f"{FEATURE_LABELS.get(prefix, feat_name)} contribution ({direction})"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Build single-row DataFrame matching training input format
        row_dict = {}
        for col in COLUMN_NAMES[:-1]:  # A1 through A15
            val = data.get(col)
            # Cast numeric types appropriately
            if col in ["A2", "A3", "A8", "A11", "A14", "A15"]:
                try:
                    row_dict[col] = float(val) if val is not None and str(val).strip() != "" else np.nan
                except (ValueError, TypeError):
                    row_dict[col] = np.nan
            else:
                row_dict[col] = str(val).strip() if val is not None and str(val).strip() != "" else np.nan
                
        input_df = pd.DataFrame([row_dict])
        
        # Transform features
        X_trans = preprocessor.transform(input_df)
        
        # Stacking ensemble prediction
        probs = ensemble.predict_proba(X_trans)[0]
        approval_prob = float(probs[1])
        rejection_prob = float(probs[0])
        
        is_approved = approval_prob >= OPTIMAL_THRESHOLD
        
        # Risk classification tier
        if approval_prob >= 0.75:
            risk_tier = "Prime / Low Risk"
            risk_color = "#10B981"  # Emerald
        elif approval_prob >= OPTIMAL_THRESHOLD:
            risk_tier = "Moderate / Near-Prime Risk"
            risk_color = "#F59E0B"  # Amber
        else:
            risk_tier = "Subprime / High Risk"
            risk_color = "#EF4444"  # Rose
            
        # SHAP calculation
        raw_shap = explainer.shap_values(X_trans)
        if isinstance(raw_shap, list):
            vals = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif getattr(raw_shap, "ndim", 0) == 3:
            vals = raw_shap[0, :, 1]
        else:
            vals = raw_shap[0]
            
        # Extract top 5 positive and top 5 negative drivers
        sorted_indices = np.argsort(np.abs(vals))[::-1]
        top_drivers = []
        for idx in sorted_indices[:8]:
            shap_val = float(vals[idx])
            feat = feature_names[idx]
            raw_input_val = str(row_dict.get(feat.split("_")[0], "N/A"))
            top_drivers.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat.split("_")[0], feat),
                "shap_value": round(shap_val, 4),
                "impact": "positive" if shap_val > 0 else "negative",
                "human_reason": explain_shap_feature(feat, X_trans[0, idx], shap_val),
                "input_value": raw_input_val,
            })
            
        return jsonify({
            "status": "success",
            "decision": "APPROVED" if is_approved else "REJECTED",
            "approval_probability": round(approval_prob * 100, 2),
            "rejection_probability": round(rejection_prob * 100, 2),
            "threshold": round(OPTIMAL_THRESHOLD * 100, 2),
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "top_drivers": top_drivers,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/sample/<sample_type>")
def get_sample(sample_type):
    """Return pre-configured applicant presets for demonstration."""
    if sample_type == "prime":
        # High approval applicant (clean credit, high assets, steady employment)
        return jsonify({
            "A1": "b", "A2": "38.5", "A3": "4.5", "A4": "u", "A5": "g",
            "A6": "q", "A7": "h", "A8": "7.5", "A9": "t", "A10": "t",
            "A11": "12", "A12": "t", "A13": "g", "A14": "120", "A15": "5000"
        })
    elif sample_type == "borderline":
        # Marginal applicant (moderate lines, modest savings, employment recent)
        return jsonify({
            "A1": "a", "A2": "24.0", "A3": "1.2", "A4": "u", "A5": "g",
            "A6": "c", "A7": "v", "A8": "0.8", "A9": "t", "A10": "f",
            "A11": "1", "A12": "f", "A13": "g", "A14": "240", "A15": "150"
        })
    elif sample_type == "subprime":
        # High default risk applicant (prior default flag, zero savings, unemployed)
        return jsonify({
            "A1": "b", "A2": "22.5", "A3": "0.5", "A4": "y", "A5": "p",
            "A6": "k", "A7": "v", "A8": "0.2", "A9": "f", "A10": "f",
            "A11": "0", "A12": "f", "A13": "g", "A14": "180", "A15": "0"
        })
    else:
        # Random sample from dataset
        sample_row = raw_df.drop(columns=[TARGET_COL]).sample(1).iloc[0]
        data = {col: str(sample_row[col]) if pd.notna(sample_row[col]) else "" for col in COLUMN_NAMES[:-1]}
        return jsonify(data)


@app.route("/api/metrics")
def get_metrics():
    """Return model comparison and fairness metrics."""
    try:
        model_cmp_path = REPORTS_DIR / "model_comparison.csv"
        fairness_path = REPORTS_DIR / "fairness_analysis.csv"
        
        cmp_df = pd.read_csv(model_cmp_path) if model_cmp_path.exists() else pd.DataFrame()
        fairness_df = pd.read_csv(fairness_path) if fairness_path.exists() else pd.DataFrame()
        
        return jsonify({
            "model_comparison": cmp_df.to_dict(orient="records"),
            "fairness_analysis": fairness_df.to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reports/<path:filename>")
def serve_report(filename):
    """Serve generated visual reports (PNGs) directly."""
    return send_from_directory(REPORTS_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Credit Underwriting Web App on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
