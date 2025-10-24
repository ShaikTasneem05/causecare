import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from dowhy import CausalModel
except Exception as e:
    print("Error importing DoWhy. Install with: pip install dowhy")
    print("Exception:", e)
    sys.exit(1)

from sklearn.preprocessing import LabelEncoder


def ensure_outputs_dir(path="outputs"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_raw(path="../data/heart.csv"):
    df = pd.read_csv(path)
    print(f"Loaded raw data from: {path}  (shape={df.shape})")
    print("Columns:", df.columns.tolist())
    return df

def quick_checks(df):
    print("\n=== Quick checks ===")
    print("Rows,Cols:", df.shape)
    print("\nHeartDisease counts:\n", df["HeartDisease"].value_counts())
    print("\nMean Cholesterol by HeartDisease:\n", df.groupby("HeartDisease")["Cholesterol"].mean())
    numeric = df.select_dtypes(include=[np.number])
    if "Cholesterol" in numeric.columns and "HeartDisease" in numeric.columns:
        print("\nPearson corr (Cholesterol vs HeartDisease):", numeric["Cholesterol"].corr(numeric["HeartDisease"]))
    else:
        print("\nSkipping Pearson corr (columns missing or non-numeric).")

def encode_categoricals(df, cols=None):
    df2 = df.copy()
    if cols is None:
        cols = df2.select_dtypes(include=['object','category']).columns.tolist()
    le = LabelEncoder()
    encoded = {}
    for col in cols:
        if col in df2.columns:
            df2[col] = le.fit_transform(df2[col].astype(str))
            encoded[col] = list(le.classes_)
            print(f"Encoded {col}: {encoded[col]}")
    return df2, encoded

def safe_str(obj):
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def run_analysis(df, treatment="Cholesterol", outcome="HeartDisease", common_causes=None):
    if common_causes is None:
        common_causes = ["Age","Sex","RestingBP","FastingBS","MaxHR","ExerciseAngina","Oldpeak","ChestPainType","RestingECG","ST_Slope"]
    common_causes = [c for c in common_causes if c in df.columns]

    print("\nUsing common_causes:", common_causes)

    model = CausalModel(data=df, treatment=treatment, outcome=outcome, common_causes=common_causes)

    identified = model.identify_effect()
    print("\nIdentified estimand:")
    print(identified)

    results = {}
    estimator_objects = {}


    estimators = {
        "backdoor.linear_regression": "linear_regression",
        "backdoor.propensity_score_matching": "psm",
        "backdoor.propensity_score_weighting": "ipw"
    }

    for method_name, label in estimators.items():
        print(f"\nEstimating ATE using {method_name} ...")
        try:
            est_obj = model.estimate_effect(identified, method_name=method_name)
            ate_val = getattr(est_obj, "value", None)
            results[label] = {
                "method_name": method_name,
                "ate": float(ate_val) if ate_val is not None else None,
                "estimator_object_str": safe_str(est_obj)
            }
            estimator_objects[label] = est_obj
            print(f"ATE ({label}):", results[label]["ate"])
        except Exception as e:
            print(f"Estimator {label} failed:", e)
            results[label] = {"method_name": method_name, "error": safe_str(e)}
            estimator_objects[label] = None

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
                
                ref_dict = {
                    "refuter": ref_method,
                    "refutation_string": safe_str(ref),
                    "new_effect": getattr(ref, "new_effect", None),
                    "p_value": getattr(ref, "p_value", None),
                    "test_type": getattr(ref, "test_type", None)
                }
                results[label]["refutations"][ref_method] = ref_dict
                print(f"Refutation ({ref_method}) result: new_effect={ref_dict['new_effect']}, p_value={ref_dict['p_value']}")
            except Exception as e:
                print(f"Refutation {ref_method} failed for {label}:", e)
                results[label]["refutations"][ref_method] = {"error": safe_str(e)}

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
            "estimator_error": r.get("error", None)
        }
        rows.append(row)
    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(outputs_dir, f"causal_results_summary_{ts}.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary to: {csv_path}")

    full = {
        "timestamp": ts,
        "identified_estimand": safe_str(identified),
        "encoded_info": encoded_info,
        "results": results
    }
    json_path = os.path.join(outputs_dir, f"causal_results_full_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, default=safe_str)
    print(f"Saved full JSON results to: {json_path}")

    return csv_path, json_path

def main():
    outputs_dir = ensure_outputs_dir("outputs")

    df_raw = load_raw("../data/heart.csv")

    quick_checks(df_raw)

    
    object_cols = [c for c in ["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"] if c in df_raw.columns]
    df_enc, encoded_info = encode_categoricals(df_raw, cols=object_cols)

    if "Cholesterol" not in df_enc.columns or "HeartDisease" not in df_enc.columns:
        print("ERROR: Required columns missing. Present columns:", df_enc.columns.tolist())
        return

    identified, results = run_analysis(df_enc,
                                       treatment="Cholesterol",
                                       outcome="HeartDisease",
                                       common_causes=["Age","Sex","RestingBP","FastingBS","MaxHR","ExerciseAngina","Oldpeak","ChestPainType","RestingECG","ST_Slope"])

    csv_path, json_path = save_results(outputs_dir, identified, encoded_info, results)

    print("\nAll done. Outputs created:")
    print(" -", csv_path)
    print(" -", json_path)

if __name__ == "__main__":
    main()
