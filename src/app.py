import streamlit as st
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
import math
import os
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
DATA_PATH = ROOT.parent / "data" / "heart_cleaned.csv"

@st.cache_resource
def load_models():
    logreg_path = MODELS_DIR / "logreg.joblib"
    rf_path = MODELS_DIR / "rf.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    feat_path = MODELS_DIR / "feature_columns.json"
    if not logreg_path.exists() or not rf_path.exists() or not feat_path.exists():
        return None, None, None, None
    try:
        logreg = joblib.load(logreg_path)
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        with open(feat_path, "r", encoding="utf-8") as f:
            feat_cols = json.load(f)
    except Exception as e:
        st.error(f"Failed to load models/artifacts: {e}")
        return None, None, None, None
    return logreg, rf, scaler, feat_cols

def latest_causal_json(outputs_dir=OUTPUTS_DIR):
    outputs_dir = Path(outputs_dir)
    files = sorted(outputs_dir.glob("causal_results_full_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def load_causal_info(json_path):
    if json_path is None:
        return None, None, None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None, None
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
        # heuristics for input types
        if "Sex" in c:
            inputs[c] = st.sidebar.selectbox("Sex", options=[0,1], index=0)
        elif "ExerciseAngina" in c or "FastingBS" in c or c.startswith("high_"):
            inputs[c] = st.sidebar.selectbox(c, options=[0,1], index=0)
        elif "ChestPainType" in c or "RestingECG" in c or "ST_Slope" in c:
            inputs[c] = st.sidebar.number_input(c, value=0, format="%d", step=1)
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

def show_causal_inference_viz(outputs_dir: Path = OUTPUTS_DIR):
    st.subheader("Causal Inference — ATE & Robustness Dashboard")
    outputs_dir = Path(outputs_dir)
    files = sorted(list(outputs_dir.glob("causal_results_full_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        st.warning("No causal JSON files found in outputs/. Run causal analysis first.")
        return
    latest = files[0]

    try:
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        st.error(f"Failed to load JSON {latest.name}: {e}")
        return

    results = data.get("results", {})
    ts = data.get("timestamp", "")
    st.caption(f"Using causal output: {latest.name}  —  timestamp: {ts}")

    rows = []
    for est_label, info in (results.items() if isinstance(results, dict) else []):
        ate = info.get("ate")
        ci = info.get("ci")
        p_placebo = None
        refuts = info.get("refutations") or {}
        if isinstance(refuts, dict):
            place = refuts.get("placebo_treatment_refuter") or refuts.get("placebo") or refuts.get("placebo_refuter")
            if isinstance(place, dict):
                p_placebo = place.get("p_value") if place.get("p_value") is not None else place.get("p")
            else:
                try:
                    txt = str(place)
                    import re
                    m = re.search(r"p value: *([0-9.eE+-]+)", txt)
                    if m:
                        p_placebo = float(m.group(1))
                except Exception:
                    pass

        ci_low, ci_high = (None, None)
        if isinstance(ci, (list,tuple)) and len(ci) >= 2:
            ci_low, ci_high = float(ci[0]), float(ci[1])
        rows.append({
            "estimator": est_label,
            "ate": (float(ate) if (ate is not None and not (isinstance(ate, str) and ate.lower() == "nan")) else None),
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "placebo_p": (float(p_placebo) if p_placebo is not None else None)
        })

    if not rows:
        st.info("No estimator results found in the JSON.")
        return

    df = pd.DataFrame(rows)

    st.markdown("**ATE comparison (with CI if available)**")
    df_plot = df.copy()
    df_plot["ate_plot"] = df_plot["ate"].fillna(0.0)

    err_y = []
    for _, r in df_plot.iterrows():
        if pd.notna(r["ci_lower"]) and pd.notna(r["ci_upper"]):
            low = r["ate_plot"] - r["ci_lower"]
            high = r["ci_upper"] - r["ate_plot"]
            err_y.append([low, high])
        else:
            val = abs(r["ate_plot"]) * 0.08 + 0.005
            err_y.append([val, val])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["estimator"],
        y=df_plot["ate_plot"],
        error_y=dict(type='data', symmetric=False, array=[e[1] for e in err_y], arrayminus=[e[0] for e in err_y]),
        marker_color='indianred'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(yaxis_title="ATE (effect on HeartDisease)", xaxis_title="Estimator", height=420)
    st.plotly_chart(fig, use_container_width=True)

    selected = st.selectbox("Choose estimator for detailed view", df_plot["estimator"].tolist())
    sel_row = df_plot[df_plot["estimator"] == selected].iloc[0]

    st.markdown(f"**Uncertainty (bootstrap-like) for `{selected}`**")
    bootstrap_vals = None
    try:
        raw_info = results.get(selected, {})
        for key in ["ates", "ate_boot", "ate_boot_samples", "bootstrap_ates", "ate_bootstrap", "ate_samples"]:
            if key in raw_info:
                bootstrap_vals = np.array(raw_info[key], dtype=float)
                break
        if bootstrap_vals is None and pd.notna(sel_row["ci_lower"]) and pd.notna(sel_row["ci_upper"]):
            mean = float(sel_row["ate_plot"])
            low = float(sel_row["ci_lower"])
            high = float(sel_row["ci_upper"])
            sd = max(1e-6, (high - low) / (2 * 1.96)) if high > low else max(1e-3, abs(mean)*0.1 + 1e-3)
            bootstrap_vals = np.random.normal(loc=mean, scale=sd, size=2000)
        elif bootstrap_vals is None:
            mean = float(sel_row["ate_plot"])
            sd = max(1e-3, abs(mean)*0.12 + 0.005)
            bootstrap_vals = np.random.normal(loc=mean, scale=sd, size=2000)
    except Exception:
        bootstrap_vals = np.random.normal(loc=float(sel_row["ate_plot"] or 0.0), scale=0.02, size=2000)

    hist_fig = px.histogram(x=bootstrap_vals, nbins=40, labels={'x':'ATE sample'}, marginal="box",
                            title=f"Bootstrap-like ATE distribution for {selected}")
    hist_fig.add_vline(x=sel_row["ate_plot"], line_color="red", annotation_text="ATE", annotation_position="top left")
    if pd.notna(sel_row["ci_lower"]) and pd.notna(sel_row["ci_upper"]):
        hist_fig.add_vline(x=sel_row["ci_lower"], line_color="green", line_dash="dot", annotation_text="CI lower", annotation_position="bottom left")
        hist_fig.add_vline(x=sel_row["ci_upper"], line_color="green", line_dash="dot", annotation_text="CI upper", annotation_position="bottom right")
    st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("**Numeric summary & refutations**")
    col1, col2 = st.columns([2,3])
    with col1:
        st.write("**Estimator**:", selected)
        st.write("ATE:", sel_row["ate"])
        st.write("CI:", (sel_row["ci_lower"], sel_row["ci_upper"]))
        st.write("Placebo p-value:", sel_row["placebo_p"])
    with col2:
        ref_table = []
        raw_info = results.get(selected, {}) or {}
        refuts = raw_info.get("refutations") or {}
        if isinstance(refuts, dict) and refuts:
            for rname, rinfo in refuts.items():
                new_eff = rinfo.get("new_effect") if isinstance(rinfo, dict) else None
                pval = rinfo.get("p_value") if isinstance(rinfo, dict) else None
                ref_table.append({"refuter": rname, "new_effect": new_eff, "p_value": pval})
            st.table(pd.DataFrame(ref_table))
        else:
            st.write("No detailed refutation entries available for this estimator.")

    st.markdown("---")

def show_3d_prediction_surface(model, scaler, feat_cols, current_inputs, data_path=DATA_PATH,
                               feat_x=None, feat_y=None, grid_size=25):
    """
    Interactive 3D surface of model predicted probability as a function of two features.
    - model: trained classifier (RandomForest recommended)
    - scaler: StandardScaler or None; if provided, will be applied to grid before predict
    - feat_cols: list of feature names (includes all features except target)
    - current_inputs: dict of current input values (as in build_input_form)
    - data_path: Path to heart_cleaned.csv to get plausible ranges
    - feat_x, feat_y: names of features to use for X/Y axes; defaults to Cholesterol_unscaled & Age
    - grid_size: resolution of grid (25x25 by default)
    """
    if model is None:
        st.warning("No predictive model available for 3D surface.")
        return

    available = [c for c in (feat_cols or []) if c != "HeartDisease"]
    default_x = "Cholesterol_unscaled" if "Cholesterol_unscaled" in available else ("Cholesterol" if "Cholesterol" in available else (available[0] if available else None))
    default_y = "Age" if "Age" in available else (available[1] if len(available) > 1 else (available[0] if available else None))
    feat_x = feat_x or default_x
    feat_y = feat_y or default_y
    if feat_x is None or feat_y is None:
        st.write("Not enough feature columns to show 3D surface.")
        return
    if feat_x == feat_y:
        st.write("Choose two different features for the 3D plot.")
        return

    st.subheader("3D Risk Surface — vary two features")
    st.caption(f"X: {feat_x} — Y: {feat_y} — other features fixed from sidebar")

    try:
        df = pd.read_csv(data_path)
    except Exception:
        df = None

    def get_range(col):
        cur_val = None
        if isinstance(current_inputs, dict):
            cur_val = current_inputs.get(col, None)
            try:
                cur_val = float(cur_val) if cur_val is not None else None
            except Exception:
                cur_val = None
        if df is not None and col in df.columns:
            series = df[col].dropna()
            if not series.empty:
                lo = float(series.quantile(0.02))
                hi = float(series.quantile(0.98))
                if cur_val is not None:
                    width = max((hi - lo) * 0.5, abs(cur_val) * 0.2 + 1e-3)
                    lo = max(lo, cur_val - width)
                    hi = min(hi, cur_val + width)
                return lo, hi
        if cur_val is not None:
            return (cur_val * 0.8, cur_val * 1.2) if abs(cur_val) > 1e-6 else (cur_val - 1.0, cur_val + 1.0)
        return -1.0, 1.0

    x_lo, x_hi = get_range(feat_x)
    y_lo, y_hi = get_range(feat_y)

    xs = np.linspace(x_lo, x_hi, grid_size)
    ys = np.linspace(y_lo, y_hi, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)

    base_row = {c: float(current_inputs.get(c, 0.0)) if current_inputs and c in current_inputs else 0.0 for c in available}

    grid_rows = []
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            r = base_row.copy()
            r[feat_x] = float(grid_x[i, j])
            r[feat_y] = float(grid_y[i, j])
            grid_rows.append([r[col] for col in available])

    X_grid = np.array(grid_rows, dtype=float)
    try:
        if scaler is not None:
            X_grid_in = scaler.transform(X_grid)
        else:
            X_grid_in = X_grid
    except Exception:
        try:
            X_grid_in = X_grid
        except Exception as e:
            st.error("Failed preparing grid for prediction: " + str(e))
            return

    try:
        probs = model.predict_proba(X_grid_in)[:, 1]
    except Exception:
        try:
            probs = model.predict(X_grid_in).astype(float)
        except Exception as e:
            st.error("Model prediction failed on grid: " + str(e))
            return

    Z = probs.reshape(grid_x.shape)

    cur_x = float(current_inputs.get(feat_x, base_row.get(feat_x, 0.0)))
    cur_y = float(current_inputs.get(feat_y, base_row.get(feat_y, 0.0)))
    cur_row = np.array([[ base_row[col] for col in available ]], dtype=float)
    try:
        cur_in = scaler.transform(cur_row) if scaler is not None else cur_row
        cur_prob = float(model.predict_proba(cur_in)[:, 1][0])
    except Exception:
        try:
            cur_prob = float(model.predict(cur_row)[0])
        except Exception:
            cur_prob = None

    surface = go.Surface(x=xs, y=ys, z=Z, colorscale="Viridis", showscale=True, name="Predicted risk")
    fig = go.Figure(data=[surface])

    if cur_prob is not None:
        fig.add_trace(go.Scatter3d(
            x=[cur_x], y=[cur_y], z=[cur_prob],
            mode='markers+text', marker=dict(size=6, color='red'),
            name='Current input', text=[f"you ({cur_prob:.3f})"], textposition="top center"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title=feat_x,
            yaxis_title=feat_y,
            zaxis_title="Predicted risk",
            aspectmode='auto'
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Heart Risk Predictor + Causal Explanation", layout="wide")
    st.title("Heart Disease Risk — Predictor + Causal Explanation")
    st.write("Predictive model + causal effect summary (Cholesterol).")

    logreg, rf, scaler, feat_cols = load_models()
    if logreg is None:
        st.error("Models not found. Run train_predictive.py first and ensure models/ contains artifacts.")
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

    try:
        prob = float(model.predict_proba(X_for_pred)[:,1][0])
        pred_class = int(prob >= 0.5)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        prob = None
        pred_class = None

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Prediction")
        if prob is not None:
            st.metric(label="Predicted risk (probability)", value=f"{prob:.3f}", delta=None)
            st.write("Predicted class (1 = disease, 0 = no disease):", pred_class)
            st.write("Note: predictive model is trained on the cleaned, preprocessed dataset. Predictions are approximate.")
        else:
            st.write("Prediction unavailable.")
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
            placebo = None
            if isinstance(refutations, dict):
                placebo = refutations.get("placebo_treatment_refuter") if isinstance(refutations.get("placebo_treatment_refuter"), dict) else None
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

    with st.expander("Causal ATE & Robustness"):
        show_causal_inference_viz(outputs_dir=OUTPUTS_DIR)

    with st.expander("3D Risk Surface (interactive)"):
        try:
            show_3d_prediction_surface(rf, scaler, feat_cols, inputs, feat_x="Cholesterol_unscaled", feat_y="Age", grid_size=30)
        except Exception as e:
            st.write("3D surface failed:", e)

    st.markdown("---")

if __name__ == "__main__":
    main()
