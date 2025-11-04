import streamlit as st
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import math
import os
from datetime import datetime

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
DATA_PATH = ROOT.parent / "data" / "heart_cleaned.csv"

def load_models():
    logreg_path = MODELS_DIR / "logreg.joblib"
    rf_path = MODELS_DIR / "rf.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    feat_path = MODELS_DIR / "feature_columns.json"
    if not logreg_path.exists() or not rf_path.exists() or not feat_path.exists():
        st.error("Models not found. Run train_predictive.py first.")
        return None, None, None, None
    logreg = joblib.load(logreg_path)
    rf = joblib.load(rf_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    with open(feat_path, "r", encoding="utf-8") as f:
        feat_cols = json.load(f)
    return logreg, rf, scaler, feat_cols

def latest_causal_json(outputs_dir=OUTPUTS_DIR):
    files = sorted(outputs_dir.glob("causal_results_full_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def load_causal_info(json_path):
    if json_path is None:
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", {})
    ident = data.get("identified_estimand", "")
    ts = data.get("timestamp", "")
    return results, ident, ts

def build_input_form(feat_cols):
    st.sidebar.header("User inputs")
    inputs = {}
    for c in feat_cols:
        if c == "HeartDisease":
            continue
        if "Sex" in c:
            inputs[c] = st.sidebar.selectbox("Sex", options=[0,1], index=0)  
        elif c.lower().startswith("age"):
            inputs[c] = st.sidebar.number_input("Age (scaled)", value=0.0, format="%.3f")
        elif "ExerciseAngina" in c or "FastingBS" in c or "high_" in c:
            inputs[c] = st.sidebar.selectbox(c, options=[0,1], index=0)
        else:
            inputs[c] = st.sidebar.number_input(c, value=0.0, format="%.4f")
    return inputs

def prepare_features_from_inputs(inputs, feat_cols):
    row = []
    for c in feat_cols:
        if c == "HeartDisease":
            continue
        v = inputs.get(c, 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        row.append(v)
    df_row = pd.DataFrame([row], columns=[c for c in feat_cols if c!="HeartDisease"])
    return df_row

def main():
    st.set_page_config(page_title="Heart Risk Predictor + Causal Explanation", layout="wide")
    st.title("Heart Disease Risk — Predictor + Causal Explanation")
    st.write("Predictive model + causal effect summary (Cholesterol).")

    logreg, rf, scaler, feat_cols = load_models()
    if logreg is None:
        st.stop()

    cj = latest_causal_json()
    causal_results, ident, ts = load_causal_info(cj) if cj else (None, None, None)

    inputs = build_input_form(feat_cols)

    X_row = prepare_features_from_inputs(inputs, feat_cols)

    model_choice = st.selectbox("Predictive model", options=["RandomForest", "LogisticRegression"])
    if model_choice == "LogisticRegression":
        if scaler is None:
            st.warning("No scaler found - logistic inputs may be off.")
            X_for_pred = X_row.values
        else:
            X_for_pred = scaler.transform(X_row)
        model = logreg
    else:
        X_for_pred = X_row.values  
        model = rf

    prob = float(model.predict_proba(X_for_pred)[:,1][0])
    pred_class = int(prob >= 0.5)

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Prediction")
        st.metric(label="Predicted risk (probability)", value=f"{prob:.3f}", delta=None)
        st.write("Predicted class (1 = disease, 0 = no disease):", pred_class)
        st.write("Note: predictive model is trained on the cleaned, preprocessed dataset. Predictions are approximate.")
    with col2:
        st.subheader("Model choice")
        st.write(f"Using: **{model_choice}**")
        st.write("Model probabilities and feature importances are approximate and depend on preprocessing.")

    st.subheader("Feature importance (RandomForest)")
    try:
        importances = rf.feature_importances_
        fi = pd.DataFrame({"feature":[c for c in feat_cols if c!="HeartDisease"], "importance": importances})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        st.dataframe(fi.head(10))
    except Exception as e:
        st.write("Could not load feature importances:", e)

    st.subheader("Causal result for Cholesterol (from DoWhy outputs)")
    if causal_results is None:
        st.write("No causal JSON found in outputs/ — run causal analysis first.")
    else:
        chosen = None
        for k in ["psm","ipw","linear_regression"]:
            if k in causal_results:
                chosen = (k, causal_results[k])
                break
        if chosen is None:
            st.write("No known estimators found in causal results.")
        else:
            label, info = chosen
            ate = info.get("ate")
            ci = info.get("ci")
            refutations = info.get("refutations", {})
            st.write(f"Estimator: **{label}**")
            st.write(f"- ATE: **{ate}**")
            if ci:
                st.write(f"- CI: {ci}")
            placebo = refutations.get("placebo_treatment_refuter") if isinstance(refutations, dict) else None
            pval = None
            if isinstance(placebo, dict):
                pval = placebo.get("p_value")
            if pval is not None:
                st.write(f"- Placebo refuter p-value: {pval}")
            st.markdown("---")
            st.write("Short interpretation:")
            if ate is None:
                st.write("ATE not available.")
            else:
                if ate > 0:
                    st.write("Higher cholesterol increases heart disease probability *on average* (according to causal analysis).")
                elif ate < 0:
                    st.write("Higher cholesterol decreases heart disease probability *on average* (according to causal analysis) — this may reflect treatment or confounding in the dataset; interpret with caution.")
                else:
                    st.write("No effect detected (ATE ≈ 0).")
    st.markdown("---")
    st.write("App generated on", datetime.now().isoformat())

if __name__ == "__main__":
    main()
