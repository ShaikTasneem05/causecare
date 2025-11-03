# quick_fix_and_run.py
"""
Quick auto-fix + causal run.
Place in src/ and run from src/ (venv active).
Reads ../data/heart_cleaned.csv, fixes common problems (unscaled cholesterol,
binary treatment), runs continuous and binary causal analyses, saves outputs
to src/outputs/.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try imports for dowhy / sklearn, else instruct user
try:
    from dowhy import CausalModel
except Exception as e:
    print("ERROR: dowhy not installed or import failed. Install with: pip install dowhy")
    raise

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

OUTDIR = Path("outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
CLEAN_PATH = Path("../data/heart_cleaned.csv")

def ensure_file():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"{CLEAN_PATH} not found. Run data_processing.py first to create cleaned CSV.")

def load_df():
    df = pd.read_csv(CLEAN_PATH)
    return df

def ensure_heartdisease_binary(df):
    # Convert common encodings to 0/1
    if "HeartDisease" not in df.columns:
        raise KeyError("HeartDisease column missing in cleaned data.")
    vals = pd.unique(df["HeartDisease"].dropna())
    # If values look like strings, try to map
    if df["HeartDisease"].dtype == object:
        mapping = {}
        # common possibilities: 'Yes'/'No', 'Y'/'N', 'Present'/'Absent'
        for v in vals:
            s = str(v).strip().lower()
            if s in ("yes","y","present","1","true","t"):
                mapping[v] = 1
            elif s in ("no","n","absent","0","false","f"):
                mapping[v] = 0
        if mapping:
            df["HeartDisease"] = df["HeartDisease"].map(mapping).astype(int)
        else:
            # try factorize and map the most frequent to 1 (less ideal)
            fac, uniq = pd.factorize(df["HeartDisease"])
            # assume the label with highest mean of other risk features corresponds to disease; fallback: map first->0 second->1
            if len(uniq) == 2:
                df["HeartDisease"] = fac
            else:
                raise ValueError("Could not automatically convert HeartDisease to binary. Values: " + str(uniq))
    else:
        # numeric: if values not 0/1, try to normalize (e.g., 1/2 -> 0/1)
        unique_vals = sorted(pd.unique(df["HeartDisease"].dropna()))
        if set(unique_vals) <= {0,1}:
            df["HeartDisease"] = df["HeartDisease"].astype(int)
        else:
            # If values are like 1 and 2, map max->1 min->0
            if len(unique_vals) == 2:
                mapping = {unique_vals[0]:0, unique_vals[1]:1}
                df["HeartDisease"] = df["HeartDisease"].map(mapping).astype(int)
            else:
                # try threshold at median
                df["HeartDisease"] = (df["HeartDisease"] > df["HeartDisease"].median()).astype(int)
    return df

def ensure_cholesterol_unscaled(df):
    # If Cholesterol_unscaled exists and looks reasonable, keep it.
    if "Cholesterol_unscaled" in df.columns:
        s = df["Cholesterol_unscaled"].dropna()
        if s.empty:
            # fallback to Cholesterol column
            if "Cholesterol" in df.columns:
                df["Cholesterol_unscaled"] = df["Cholesterol"].copy()
        else:
            # sanity check range
            if (s.min() < 0) or (s.mean() < 10):  # suspicious small numbers
                # fallback to Cholesterol column if exists
                if "Cholesterol" in df.columns:
                    df["Cholesterol_unscaled"] = df["Cholesterol"].copy()
    else:
        # try to create from Cholesterol
        if "Cholesterol" in df.columns:
            df["Cholesterol_unscaled"] = df["Cholesterol"].copy()
        else:
            raise KeyError("No Cholesterol column found to create Cholesterol_unscaled.")
    return df

def choose_threshold(df):
    # Choose clinical threshold 240 if data mean seems like mg/dL (50 < mean < 500)
    mean_ch = df["Cholesterol_unscaled"].dropna().mean()
    if 50 < mean_ch < 500:
        threshold = 240
        used = "clinical_240"
    else:
        threshold = df["Cholesterol_unscaled"].median()
        used = "median"
    return threshold, used

def make_binary_treatment(df, threshold):
    df["high_cholesterol"] = (df["Cholesterol_unscaled"] >= threshold).astype(int)
    return df

def quick_predictive_check(df):
    # simple logistic on Cholesterol_unscaled
    X = df[["Cholesterol_unscaled"]].copy().fillna(df["Cholesterol_unscaled"].median()).values.reshape(-1,1)
    y = df["HeartDisease"].values
    try:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
        clf = LogisticRegression(solver="liblinear").fit(X_train, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        coef = float(clf.coef_[0][0])
        return {"coef": coef, "auc": float(auc)}
    except Exception as e:
        return {"error": str(e)}

# helper parse refutation text
def parse_refutation_text(txt):
    m_new = re.search(r"New effect: *([-\d.eE]+)", str(txt))
    m_p = re.search(r"p value: *([-\d.eE]+)", str(txt))
    new_effect = float(m_new.group(1)) if m_new else None
    p_value = float(m_p.group(1)) if m_p else None
    return new_effect, p_value

# simple bootstrap for linear regression ATE
def bootstrap_ate(df, treatment, outcome, confounders, n_boot=200):
    ates = []
    for i in range(n_boot):
        sample = df.sample(frac=1.0, replace=True)
        try:
            model = CausalModel(data=sample, treatment=treatment, outcome=outcome, common_causes=confounders)
            ident = model.identify_effect()
            est = model.estimate_effect(ident, method_name="backdoor.linear_regression")
            ates.append(getattr(est, "value", np.nan))
        except Exception:
            ates.append(np.nan)
    arr = np.array(ates, dtype=float)
    mean = float(np.nanmean(arr))
    lo = float(np.nanpercentile(arr, 2.5))
    hi = float(np.nanpercentile(arr, 97.5))
    return mean, lo, hi

def run_single_analysis(df, treatment_col, outputs_dir, use_binary=False):
    outcome = "HeartDisease"
    common_causes = ["Age","Sex","RestingBP","FastingBS","MaxHR","ExerciseAngina","Oldpeak","ChestPainType","RestingECG","ST_Slope"]
    common_causes = [c for c in common_causes if c in df.columns]

    model = CausalModel(data=df, treatment=treatment_col, outcome=outcome, common_causes=common_causes)
    identified = model.identify_effect()

    # choose estimators
    estimators = {"backdoor.linear_regression":"linear_regression"}
    if use_binary:
        estimators.update({
            "backdoor.propensity_score_matching":"psm",
            "backdoor.propensity_score_weighting":"ipw"
        })

    results = {}
    estimator_objs = {}
    for method_name, label in estimators.items():
        try:
            est = model.estimate_effect(identified, method_name=method_name)
            val = getattr(est, "value", None)
            results[label] = {
                "method_name": method_name,
                "ate": float(val) if val is not None else None,
                "estimator_object_str": str(est)
            }
            estimator_objs[label] = est
        except Exception as e:
            results[label] = {"method_name": method_name, "error": str(e)}
            estimator_objs[label] = None

    # bootstrap only for linear_regression if it succeeded
    if estimator_objs.get("linear_regression") is not None:
        try:
            mean, lo, hi = bootstrap_ate(df, treatment_col, outcome, common_causes, n_boot=150)
            results["linear_regression"]["ci"] = [lo, hi]
            results["linear_regression"]["ate_boot_mean"] = mean
        except Exception as e:
            results["linear_regression"]["ci_error"] = str(e)

    # refutations
    ref_methods = ["random_common_cause","placebo_treatment_refuter","data_subset_refuter"]
    for label, obj in estimator_objs.items():
        if obj is None:
            continue
        results[label]["refutations"] = {}
        for r in ref_methods:
            try:
                ref = model.refute_estimate(identified, obj, method_name=r)
                txt = str(ref)
                new_effect, pval = parse_refutation_text(txt)
                results[label]["refutations"][r] = {
                    "refuter": r,
                    "refutation_string": txt,
                    "new_effect": new_effect,
                    "p_value": pval
                }
            except Exception as e:
                results[label]["refutations"][r] = {"error": str(e)}
    return identified, results

def save_outputs(prefix, identified, results, encoded_info=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_full = {
        "prefix": prefix,
        "timestamp": ts,
        "identified_estimand": str(identified),
        "results": results,
        "encoded_info": encoded_info or {}
    }
    # file names
    json_path = OUTDIR / f"causal_results_full_{prefix}_{ts}.json"
    csv_path = OUTDIR / f"causal_results_summary_{prefix}_{ts}.csv"

    # save structured csv summary (one row per estimator)
    rows = []
    for label, info in results.items():
        row = {
            "prefix": prefix,
            "timestamp": ts,
            "estimator_label": label,
            "method_name": info.get("method_name"),
            "ate": info.get("ate"),
            "ci_lower": None,
            "ci_upper": None,
            "error": info.get("error", None)
        }
        ci = info.get("ci")
        if ci and isinstance(ci, (list,tuple)) and len(ci) >= 2:
            row["ci_lower"], row["ci_upper"] = ci[0], ci[1]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_full, f, indent=2, default=str)

    return str(json_path), str(csv_path)

def make_simple_plots(df, prefix):
    # HeartDisease rate by cholesterol bins
    try:
        df2 = df.dropna(subset=["Cholesterol_unscaled","HeartDisease"]).copy()
        df2["chol_bin"] = pd.cut(df2["Cholesterol_unscaled"], bins=10)
        grp = df2.groupby("chol_bin")["HeartDisease"].mean()
        fig, ax = plt.subplots(figsize=(8,4))
        grp.plot(marker='o', ax=ax)
        ax.set_ylabel("Mean HeartDisease")
        ax.set_xlabel("Cholesterol bins")
        ax.set_title(f"HeartDisease rate by Cholesterol bins ({prefix})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        p = OUTDIR / f"{prefix}_heartd_rate_by_chol.png"
        fig.savefig(p)
        plt.close(fig)
        return str(p)
    except Exception as e:
        return None

def main():
    ensure_file()
    df = load_df()
    print("Loaded cleaned data shape:", df.shape)

    # 1) ensure HeartDisease binary
    df = ensure_heartdisease_binary(df)
    print("HeartDisease unique after fix:", sorted(df["HeartDisease"].unique()))

    # 2) ensure Cholesterol_unscaled exists and looks okay
    df = ensure_cholesterol_unscaled(df)
    print("Cholesterol_unscaled stats:\n", df["Cholesterol_unscaled"].describe())

    # 3) make binary treatment using clinical or median threshold
    threshold, used = choose_threshold(df)
    df = make_binary_treatment(df, threshold)
    print(f"Created high_cholesterol with threshold {threshold} ({used})")
    print("high_cholesterol value counts:", df["high_cholesterol"].value_counts())

    # 4) quick predictive check
    pred = quick_predictive_check(df)
    print("Predictive check (logistic on cholesterol):", pred)

    # 5) Run continuous treatment analysis (Cholesterol_unscaled)
    print("\nRunning continuous treatment analysis (Cholesterol_unscaled)...")
    ident_c, res_c = run_single_analysis(df, "Cholesterol_unscaled", OUTDIR, use_binary=False)
    json_c, csv_c = save_outputs("continuous", ident_c, res_c)
    print("Saved continuous outputs:", json_c, csv_c)
    plotc = make_simple_plots(df, "continuous")
    print("Saved continuous plot:", plotc)

    # 6) Run binary treatment analysis (high_cholesterol)
    print("\nRunning binary treatment analysis (high_cholesterol)...")
    ident_b, res_b = run_single_analysis(df, "high_cholesterol", OUTDIR, use_binary=True)
    json_b, csv_b = save_outputs("binary", ident_b, res_b)
    print("Saved binary outputs:", json_b, csv_b)
    plotb = make_simple_plots(df, "binary")
    print("Saved binary plot:", plotb)

    # 7) Create a simple human-readable summary
    summary_lines = []
    summary_lines.append("# Quick Causal Run Summary")
    summary_lines.append("Generated: " + datetime.now().isoformat())
    summary_lines.append("\n## Predictive check (logistic on Cholesterol_unscaled)")
    summary_lines.append(str(pred))

    def add_result_block(prefix, results):
        summary_lines.append(f"\n## Results for {prefix}\n")
        for label, info in results.items():
            summary_lines.append(f"- Estimator: {label}")
            summary_lines.append(f"  - method: {info.get('method_name')}")
            summary_lines.append(f"  - ATE: {info.get('ate')}")
            if info.get("ci"):
                summary_lines.append(f"  - CI: {info.get('ci')}")
            if info.get("refutations"):
                for rname, r in info.get("refutations").items():
                    summary_lines.append(f"    - Refuter {rname}: new_effect={r.get('new_effect')}, p={r.get('p_value')}")
            if info.get("error"):
                summary_lines.append(f"  - Error: {info.get('error')}")
            summary_lines.append("")

    add_result_block("continuous", res_c)
    add_result_block("binary", res_b)
    summary_path = OUTDIR / "quick_causal_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("Quick summary saved to:", summary_path)
    print("All done. Check outputs directory for JSON/CSV/plots/summary.")

if __name__ == "__main__":
    main()
