# causal_analysis.py
import os
import sys
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    from dowhy import CausalModel
except Exception as e:
    print("Error importing DoWhy. Install with: pip install dowhy")
    print("Exception:", e)
    sys.exit(1)


def ensure_outputs_dir(path="outputs"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_cleaned(path="../data/heart_cleaned.csv"):
    df = pd.read_csv(path)
    print(f"Loaded cleaned data from: {path}  (shape={df.shape})")
    return df

def quick_checks(df):
    print("\n=== Quick checks ===")
    print("Rows,Cols:", df.shape)
    if "HeartDisease" in df.columns:
        print("\nHeartDisease counts:\n", df["HeartDisease"].value_counts())
    if "Cholesterol_unscaled" in df.columns:
        print("\nCholesterol_unscaled stats:\n", df["Cholesterol_unscaled"].describe())

def parse_refutation_text(txt):
    """
    Extract numeric new_effect and p_value from DoWhy refutation text output.
    Returns (new_effect (float|None), p_value (float|None))
    """
    if txt is None:
        return None, None
    # often text contains: "New effect:XXX" and "p value:YYY"
    m_new = re.search(r"New effect: *([-\d.eE]+)", str(txt))
    m_p = re.search(r"p value: *([-\d.eE]+)", str(txt))
    new_effect = float(m_new.group(1)) if m_new else None
    p_value = float(m_p.group(1)) if m_p else None
    return new_effect, p_value

def bootstrap_ate(df, treatment, outcome, common_causes, n_boot=200):
    ates = []
    for i in range(n_boot):
        sample = df.sample(frac=1.0, replace=True)
        try:
            model = CausalModel(data=sample, treatment=treatment, outcome=outcome, common_causes=common_causes)
            identified = model.identify_effect()
            est = model.estimate_effect(identified, method_name="backdoor.linear_regression")
            ates.append(getattr(est, "value", np.nan))
        except Exception:
            ates.append(np.nan)
    ates = np.array(ates, dtype=float)
    lower = np.nanpercentile(ates, 2.5)
    upper = np.nanpercentile(ates, 97.5)
    mean = np.nanmean(ates)
    return mean, lower, upper

def run_analysis(df, outputs_dir, treatment_col="Cholesterol_unscaled", use_binary=False):
    """
    Run DoWhy causal analysis on dataframe `df`.
    - outputs_dir: path (string) where CSV/JSON will be saved
    - treatment_col: column name for treatment (continuous or pre-created binary)
    - use_binary: if True, create 'high_cholesterol' (median split) and run PSM/IPW too
    Returns: identified, results dict
    """
    # Choose treatment column and prepare binary if requested
    if treatment_col not in df.columns and not use_binary:
        raise ValueError(f"Treatment column {treatment_col} not found in dataframe.")

    if use_binary:
        # create binary treatment using median threshold (you can change to clinical threshold)
        binary_col = "high_cholesterol"
        if binary_col not in df.columns:
            df[binary_col] = (df["Cholesterol_unscaled"] >= df["Cholesterol_unscaled"].median()).astype(int)
        treatment = binary_col
        print(f"Using binary treatment: {treatment} (median split)")
    else:
        treatment = treatment_col
        print(f"Using continuous treatment: {treatment}")

    outcome = "HeartDisease"
    common_causes = ["Age","Sex","RestingBP","FastingBS","MaxHR","ExerciseAngina","Oldpeak","ChestPainType","RestingECG","ST_Slope"]
    common_causes = [c for c in common_causes if c in df.columns]

    print("\nUsing common_causes:", common_causes)

    model = CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=common_causes)
    identified = model.identify_effect()
    print("\nIdentified estimand:")
    print(identified)

    results = {}
    estimator_objects = {}

    # Only run PSM/IPW if treatment is binary
    estimators_continuous = {
        "backdoor.linear_regression": "linear_regression"
    }
    estimators_binary_extra = {
        "backdoor.propensity_score_matching": "psm",
        "backdoor.propensity_score_weighting": "ipw"
    }

    # merge dicts depending on treatment type
    estimators = estimators_continuous.copy()
    if use_binary:
        estimators.update(estimators_binary_extra)

    for method_name, label in estimators.items():
        print(f"\nEstimating ATE using {method_name} ...")
        try:
            est_obj = model.estimate_effect(identified, method_name=method_name)
            ate_val = getattr(est_obj, "value", None)
            results[label] = {
                "method_name": method_name,
                "ate": float(ate_val) if ate_val is not None else None,
                "estimator_object_str": str(est_obj)
            }
            estimator_objects[label] = est_obj
            print(f"ATE ({label}):", results[label]["ate"])
        except Exception as e:
            print(f"Estimator {label} failed:", e)
            results[label] = {"method_name": method_name, "error": str(e)}
            estimator_objects[label] = None

    # Add bootstrap CI for linear_regression (if it ran)
    if estimator_objects.get("linear_regression") is not None:
        print("\nComputing bootstrap CI for linear_regression (this may take a bit)...")
        try:
            mean, lower, upper = bootstrap_ate(df, treatment, outcome, common_causes, n_boot=200)
            results["linear_regression"]["ci"] = [float(lower), float(upper)]
            results["linear_regression"]["ate_boot_mean"] = float(mean)
            print("Bootstrap CI:", lower, upper)
        except Exception as e:
            print("Bootstrap failed:", e)

    # Refutations
    refutation_methods = [
        "random_common_cause",
        "placebo_treatment_refuter",
        "data_subset_refuter"
    ]

    for label, est_info in list(results.items()):
        est_obj = estimator_objects.get(label)
        if est_obj is None:
            print(f"\nSkipping refutations for {label} (estimator object not available).")
            continue

        results[label]["refutations"] = {}
        for ref_method in refutation_methods:
            print(f"\nRunning refutation '{ref_method}' for estimator {label} ...")
            try:
                ref = model.refute_estimate(identified, est_obj, method_name=ref_method)
                # parse textual refutation (extract numeric fields where available)
                ref_text = str(ref)  # text blob
                new_effect, p_value = parse_refutation_text(ref_text)
                # try to also get attributes directly if present
                ref_dict = {
                    "refuter": ref_method,
                    "refutation_string": ref_text,
                    "new_effect": float(new_effect) if new_effect is not None else None,
                    "p_value": float(p_value) if p_value is not None else None,
                    "test_type": getattr(ref, "test_type", None) if hasattr(ref, "test_type") else None
                }
                results[label]["refutations"][ref_method] = ref_dict
                print(f"Refutation ({ref_method}) result: new_effect={ref_dict['new_effect']}, p_value={ref_dict['p_value']}")
            except Exception as e:
                print(f"Refutation {ref_method} failed for {label}:", e)
                results[label]["refutations"][ref_method] = {"error": str(e)}

    return identified, results

def save_results(outputs_dir, identified, encoded_info, results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for label, r in results.items():
        row = {
            "timestamp": ts,
            "estimator_label": label,
            "method_name": r.get("method_name"),
            "ate": r.get("ate"),
            "ci_lower": None,
            "ci_upper": None,
            "estimator_error": r.get("error", None)
        }
        if r.get("ci"):
            row["ci_lower"], row["ci_upper"] = r.get("ci")
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(outputs_dir, f"causal_results_summary_{ts}.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary to: {csv_path}")

    full = {
        "timestamp": ts,
        "identified_estimand": str(identified),
        "encoded_info": encoded_info,
        "results": results
    }
    json_path = os.path.join(outputs_dir, f"causal_results_full_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, default=str)
    print(f"Saved full JSON results to: {json_path}")

    return csv_path, json_path

# --- New convenience function so other scripts can import this module ---
def run_file_pipeline(cleaned_csv_path="../data/heart_cleaned.csv", outputs_dir="outputs",
                      treatment_col="Cholesterol_unscaled", use_binary=False):
    """
    Run the full pipeline from cleaned CSV to saved outputs.
    Returns: (csv_path, json_path)
    """
    outputs_dir = ensure_outputs_dir(outputs_dir)
    df = load_cleaned(cleaned_csv_path)
    quick_checks(df)

    # minimal encoding if needed
    object_cols = [c for c in ["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"] if c in df.columns and df[c].dtype == object]
    encoded_info = {}
    if object_cols:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for c in object_cols:
            df[c] = le.fit_transform(df[c].astype(str))
            encoded_info[c] = list(le.classes_)

    identified, results = run_analysis(df, outputs_dir, treatment_col=treatment_col, use_binary=use_binary)
    csv_path, json_path = save_results(outputs_dir, identified, encoded_info, results)
    return csv_path, json_path

def main():
    # Default behavior: read ../data/heart_cleaned.csv, write to src/outputs/, use unscaled continuous treatment.
    outputs_dir = ensure_outputs_dir("outputs")  # this 'outputs' is inside src/

    # CLI support for simple flags:
    # --binary to run binary treatment (creates high_cholesterol and runs PSM/IPW)
    # --treatment <colname> to specify treatment column
    use_binary = False
    treatment_col = "Cholesterol_unscaled"
    args = sys.argv[1:]
    if "--binary" in args:
        use_binary = True
    if "--treatment" in args:
        idx = args.index("--treatment")
        if idx + 1 < len(args):
            treatment_col = args[idx + 1]

    identified, results = run_analysis(load_cleaned("../data/heart_cleaned.csv"), outputs_dir,
                                       treatment_col=treatment_col, use_binary=use_binary)

    # minimal encoded_info detection for saving
    encoded_info = {}
    csv_path, json_path = save_results(outputs_dir, identified, encoded_info, results)

    print("\nAll done. Outputs created:")
    print(" -", csv_path)
    print(" -", json_path)

if __name__ == "__main__":
    main()
